import torch
from utils.beta_distribution import Beta
import math
import gc
import random

def scale_pKi(pKi_values, min_pKi=0.0, max_pKi=12.0):
    # Clip values to the range [min_pKi, max_pKi]
    clipped_values = torch.clamp(pKi_values, min_pKi, max_pKi)
    
    # Scale to [0, 1]
    scaled_values = (clipped_values - min_pKi) / (max_pKi - min_pKi)
    
    return scaled_values

def beta_loss_with_mixed_targets(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    reg_targets: torch.Tensor,
    class_targets: torch.Tensor,
    min_pKi: float = 0.0,
    max_pKi: float = 12.0,
    threshold: float = 4.0,
    source_labels: list = None,
    source_priors: dict = None,
    regul_lambda: float = 1.0,
    return_mean: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    alpha = torch.clamp(alpha, min=eps)
    beta = torch.clamp(beta, min=eps)
    reg_targets = torch.as_tensor(reg_targets, dtype=alpha.dtype, device=alpha.device)
    class_targets = torch.as_tensor(class_targets, dtype=alpha.dtype, device=alpha.device)
    
    scaled_threshold = (threshold - min_pKi) / (max_pKi - min_pKi)
    scaled_reg_targets = scale_pKi(reg_targets, min_pKi, max_pKi)
    
    loss = torch.zeros_like(alpha)
    reg_mask = ~torch.isnan(reg_targets)
    
    if reg_mask.any():
        reg_distribution = Beta(alpha[reg_mask], beta[reg_mask])
        loss[reg_mask] = -reg_distribution.log_prob(
            scaled_reg_targets[reg_mask].clamp(eps, 1-eps)
        )

    class_mask = ~reg_mask
    if class_mask.any():
        class_distribution = Beta(alpha[class_mask], beta[class_mask])
        class_probs = class_distribution.cdf(torch.tensor(scaled_threshold, device=alpha.device))
        
        high_mask = (class_targets == 1) & class_mask
        if high_mask.any():
            loss[high_mask] = -torch.log(1 - class_probs[high_mask[class_mask]] + eps)
        
        low_mask = (class_targets == 0) & class_mask
        if low_mask.any():
            loss[low_mask] = -torch.log(class_probs[low_mask[class_mask]] + eps)
    
    if source_labels is not None and source_priors is not None:
        phi = alpha + beta
        
        # Check if all sources are NaN (handling both float nan and string 'nan')
        if all((isinstance(source, float) and math.isnan(source)) or 
               (isinstance(source, str) and source.lower() == 'nan') 
               for source in source_labels):
            # print("WARNING: All source labels are NaN. Skipping prior loss calculation.")
            return loss.mean() if return_mean else loss
        

        max_phi = torch.tensor([source_priors[source]["max_phi"] for source in source_labels],
                             device=alpha.device)
        
        # Log-space ReLU penalty for excess concentration
        # prior_loss = torch.relu(torch.log(phi + eps) - torch.log(max_phi + eps))
        
        # Calculate normalized excess and apply quadratic penalty
        normalized_excess = (phi - max_phi) / max_phi
        prior_loss = 0.5 * torch.square(torch.relu(normalized_excess))
        
        loss = loss + regul_lambda * prior_loss

    return loss.mean() if return_mean else loss


def normal_loss_with_mixed_targets(loc, scale, reg_targets, class_targets, threshold=4, min_pKi=0, max_pKi=12):
    eps = 1e-6
    # Ensure scale is positive
    scale = torch.clamp(scale, min=eps)
    
    # Ensure all inputs are tensors
    reg_targets = torch.as_tensor(reg_targets, dtype=loc.dtype, device=loc.device)
    class_targets = torch.as_tensor(class_targets, dtype=loc.dtype, device=loc.device)
    
    # Initialize loss tensor
    loss = torch.zeros_like(loc)
    
    # Mask for regression targets (where reg_targets is not NaN)
    reg_mask = ~torch.isnan(reg_targets)
    
    # Regression loss
    if reg_mask.any():
        reg_distribution = torch.distributions.Normal(loc[reg_mask], scale[reg_mask])
        reg_loss = -reg_distribution.log_prob(reg_targets[reg_mask])
        loss[reg_mask] = reg_loss

    # Classification loss
    class_mask = ~reg_mask
    if class_mask.any():
        class_distribution = torch.distributions.Normal(loc[class_mask], scale[class_mask])
        
        # Calculate CDFs at both threshold and max_pKi for proper normalization
        cdf_threshold = class_distribution.cdf(torch.tensor(threshold, device=loc.device))
        cdf_max = class_distribution.cdf(torch.tensor(max_pKi, device=loc.device))
        
        # For class target 1 (threshold <= pKi <= max_pKi)
        high_mask = (class_targets == 1) & class_mask
        if high_mask.any():
            # Probability mass between threshold and max_pKi
            prob_between = (cdf_max[high_mask[class_mask]] - cdf_threshold[high_mask[class_mask]])
            loss[high_mask] = -torch.log(prob_between + eps)
        
        # For class target 0 (min_pKi <= pKi < threshold)
        low_mask = (class_targets == 0) & class_mask
        if low_mask.any():
            # Probability mass between min_pKi and threshold
            prob_between = (cdf_threshold[low_mask[class_mask]] - cdf_min[low_mask[class_mask]])
            loss[low_mask] = -torch.log(prob_between + eps)
    
    return loss.mean()

import torch
import torch.nn.functional as F

def multi_label_bce_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """
    Calculates Binary Cross-Entropy loss for multi-label classification,
    handling missing labels indicated by -1.

    Args:
        pred (torch.Tensor): Predicted logits from the model. 
                             Shape: (batch_size, num_classes)
        true (torch.Tensor): Ground truth labels. Shape: (batch_size, num_classes).
                             Contains 0, 1, or -1 (for missing/masked labels).

    Returns:
        torch.Tensor: A scalar tensor representing the average loss across 
                      valid classes in the batch. Returns 0.0 if no 
                      valid labels are present in the batch for any class.
    """
    # Ensure true labels are float, as BCEWithLogitsLoss expects float targets
    true = true.float() 
    
    # --- Step 1: Create mask for valid labels (0 or 1) ---
    # Shape: (batch_size, num_classes)
    valid_mask = (true != -1).float() 

    # --- Step 2: Calculate element-wise BCEWithLogitsLoss ---
    # Using reduction='none' to get loss per element
    # This calculates loss even for invalid positions, but we'll mask them out.
    # Using BCEWithLogitsLoss is numerically more stable than Sigmoid + BCELoss.
    # Shape: (batch_size, num_classes)
    element_loss = F.binary_cross_entropy_with_logits(pred, true, reduction='none')

    # --- Step 3: Apply the mask to zero out losses for invalid labels ---
    # Shape: (batch_size, num_classes)
    masked_loss = element_loss * valid_mask

    # --- Step 4: Calculate the sum of losses per class ---
    # Sum losses only where the mask is 1
    # Shape: (num_classes,)
    class_loss_sum = masked_loss.sum(dim=0)

    # --- Step 5: Count the number of valid labels per class ---
    # Shape: (num_classes,)
    class_valid_count = valid_mask.sum(dim=0)
    
    # --- Step 6: Identify classes that had at least one valid label in the batch ---
    # If class_valid_count > 0 for a class, it means we should include its loss.
    # Shape: (num_classes,) - boolean tensor
    class_is_valid_in_batch = class_valid_count > 0

    # --- Step 7: Calculate average loss *per valid class* ---
    # Avoid division by zero for classes with no valid entries by adding a small epsilon.
    # The result for classes with count 0 doesn't matter as they'll be filtered out.
    avg_loss_per_valid_class = class_loss_sum / (class_valid_count + 1e-8) 

    # --- Step 8: Average the losses only across classes that were valid in this batch ---
    num_valid_classes_in_batch = class_is_valid_in_batch.sum()

    if num_valid_classes_in_batch == 0:
        # Handle the edge case where no class had any valid labels in this batch
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    else:
        # Select the average losses only from the valid classes and compute the final mean
        final_loss = avg_loss_per_valid_class[class_is_valid_in_batch].mean()
        return final_loss


class LearningManager:
    def __init__(self, total_iterations, device, 
                 warmup_iterations=1000, 
                 mutation_start_scaling=2000, 
                 mutation_end_scaling=4000,
                 lrate=1e-3, min_lrate=1e-4, lr_decay_iters=600000,
                 initial_accumulation_steps=1, final_accumulation_steps=8, 
                 start_accumulation_ramp=100000,
                 accumulation_ramp_iters=400000,
                 # --- Old static weights (kept for potential backward compatibility or simple use) ---
                 regression_weight=1.0, 
                 multiclassification_weight=1.0,
                 # --- New dynamic weight parameters ---
                 initial_regression_weight=None, final_regression_weight=None,
                 regression_weight_start_iter=0, regression_weight_end_iter=None,
                 initial_multiclassification_weight=None, final_multiclassification_weight=None,
                 multiclassification_weight_start_iter=0, multiclassification_weight_end_iter=None,
                 # --- Other params ---
                 regularization_lambda=1.0,
                 source_priors=None):
        
        self.total_iterations = total_iterations
        self.device = device
        
        # Curriculum learning parameters
        self.warmup_iterations = warmup_iterations
        self.mutation_start_scaling = mutation_start_scaling
        self.mutation_end_scaling = mutation_end_scaling
        
        # Learning rate parameters
        self.lrate = lrate
        self.min_lrate = min_lrate
        self.lr_decay_iters = lr_decay_iters
        
        # Gradient accumulation parameters
        self.initial_accumulation_steps = initial_accumulation_steps
        self.final_accumulation_steps = final_accumulation_steps
        
        self.start_accumulation_ramp = start_accumulation_ramp
        self.accumulation_ramp_iters = accumulation_ramp_iters
        
        # --- Configure Loss Weights ---
        # Prioritize new dynamic parameters if provided, otherwise use static ones.
        self.initial_regression_weight = initial_regression_weight if initial_regression_weight is not None else regression_weight
        self.final_regression_weight = final_regression_weight if final_regression_weight is not None else regression_weight
        self.regression_weight_start_iter = regression_weight_start_iter
        # Default end_iter to total_iterations if not specified, or if start==end
        self.regression_weight_end_iter = regression_weight_end_iter if regression_weight_end_iter is not None else total_iterations
        if self.regression_weight_end_iter <= self.regression_weight_start_iter:
             self.regression_weight_end_iter = self.regression_weight_start_iter # Treat as step or constant

        self.initial_multiclassification_weight = initial_multiclassification_weight if initial_multiclassification_weight is not None else multiclassification_weight
        self.final_multiclassification_weight = final_multiclassification_weight if final_multiclassification_weight is not None else multiclassification_weight
        self.multiclassification_weight_start_iter = multiclassification_weight_start_iter
        self.multiclassification_weight_end_iter = multiclassification_weight_end_iter if multiclassification_weight_end_iter is not None else total_iterations
        if self.multiclassification_weight_end_iter <= self.multiclassification_weight_start_iter:
            self.multiclassification_weight_end_iter = self.multiclassification_weight_start_iter # Treat as step or constant

        # Regularization
        self.regularization_lambda = regularization_lambda

        # Loss functions
        self.regression_loss_fn = beta_loss_with_mixed_targets # Renamed to avoid conflict
        self.mclassification_loss_fn = multi_label_bce_loss  # Renamed to avoid conflict

        self.source_priors = source_priors 
    
    
    # --- Helper for linear ramp calculation ---
    def _get_linear_ramp_value(self, iter_num, start_iter, end_iter, initial_val, final_val):
        if initial_val == final_val or start_iter == end_iter:
            return final_val # Constant or step function case
            
        if iter_num < start_iter:
            return initial_val
        elif iter_num >= end_iter:
            return final_val
        else:
            progress = (iter_num - start_iter) / (end_iter - start_iter)
            return initial_val + progress * (final_val - initial_val)

    # --- Methods to get current loss weights ---
    def get_current_regression_weight(self, iter_num):
        return self._get_linear_ramp_value(
            iter_num,
            self.regression_weight_start_iter,
            self.regression_weight_end_iter,
            self.initial_regression_weight,
            self.final_regression_weight
        )

    def get_current_multiclassification_weight(self, iter_num):
        return self._get_linear_ramp_value(
            iter_num,
            self.multiclassification_weight_start_iter,
            self.multiclassification_weight_end_iter,
            self.initial_multiclassification_weight,
            self.final_multiclassification_weight
        )

    def get_loss_scale(self, iter_num, start_iter, end_iter, max_val=1.0):
        if iter_num < start_iter:
            return 0.0
        
        elif iter_num < end_iter:
            return (iter_num - start_iter) / (end_iter - start_iter) * max_val
        
        return max_val

    def get_proba_scale(self, iter_num, start_iter, end_iter, max_val=1.0):
        if iter_num < start_iter:
            return 0.0
        
        elif iter_num < end_iter:
            return (iter_num - start_iter) / (end_iter - start_iter) * max_val
        
        return max_val

    def use_wild_only(self, iter_num):
        return iter_num <= self.mutation_start_scaling

    def get_lr(self, iter):
        if iter < self.warmup_iterations:
            return self.lrate * iter / self.warmup_iterations
        if iter > self.lr_decay_iters:
            return self.min_lrate
        # Linear decay
        decay_ratio = (iter - self.warmup_iterations) / (self.lr_decay_iters - self.warmup_iterations)
        assert 0 <= decay_ratio <= 1
        # Linear interpolation between lrate and min_lrate
        return self.lrate + decay_ratio * (self.min_lrate - self.lrate)

    def get_current_accumulation_steps(self, iter_num):
        # Use the _get_linear_ramp_value helper for consistency
        # Note: accumulation steps should be integers
        current_steps = self._get_linear_ramp_value(
            iter_num,
            self.start_accumulation_ramp,
            self.start_accumulation_ramp + self.accumulation_ramp_iters,
            self.initial_accumulation_steps,
            self.final_accumulation_steps
        )
        return int(round(current_steps)) # Round to nearest integer
    
    # --- Update compute_loss methods to use current weights ---
    def compute_loss(self, data, reg_alpha, reg_beta, mcls_pred, pocket_loss, o_loss, cl_loss, iter_num): # Added iter_num
        loss_val = torch.tensor(0.).to(self.device)
        # Assuming pocket_loss, o_loss, cl_loss are pre-scaled or don't need scaling here
        loss_val += pocket_loss + o_loss + cl_loss 

        # Get current weights
        current_reg_weight = self.get_current_regression_weight(iter_num)
        current_mcls_weight = self.get_current_multiclassification_weight(iter_num)

        reg_loss = torch.tensor(0.).to(self.device) # Initialize as tensor
        if reg_alpha is not None and reg_beta is not None and current_reg_weight > 0:
            # Calculate raw loss
            raw_reg_loss = self.regression_loss_fn(reg_alpha, reg_beta, data.reg_y.squeeze(), data.cls_y.squeeze(), min_pKi=0, max_pKi=12, threshold=4,
                                           source_labels=data.source_y if hasattr(data, 'source_y') else None, 
                                           source_priors=self.source_priors,
                                           regul_lambda=self.regularization_lambda)
            # Apply dynamic weight
            reg_loss = raw_reg_loss * current_reg_weight
            loss_val += reg_loss

        mcls_loss = torch.tensor(0.).to(self.device) # Initialize as tensor
        if mcls_pred is not None and current_mcls_weight > 0:
            mcls_y = data.mcls_y
            # batch_size = mcls_pred.shape[0] # Get batch size from prediction tensor
            # num_classes = mcls_pred.shape[1] # Get num classes from prediction tensor
            # mcls_y = mcls_y.view(batch_size, num_classes)

            # Calculate raw loss
            raw_mcls_loss = self.mclassification_loss_fn(mcls_pred, mcls_y) 
            # Apply dynamic weight
            mcls_loss = raw_mcls_loss * current_mcls_weight
            loss_val += mcls_loss

        # Return total loss and individual *weighted* losses
        return loss_val, reg_loss, mcls_loss


    def compute_prediction_loss(self, data, reg_alpha, reg_beta, mcls_pred, iter_num): # Added iter_num
        loss_val = torch.tensor(0.).to(self.device)

        # Get current weights
        current_reg_weight = self.get_current_regression_weight(iter_num)
        current_mcls_weight = self.get_current_multiclassification_weight(iter_num)

        reg_loss = torch.tensor(0.).to(self.device) # Initialize as tensor
        if reg_alpha is not None and reg_beta is not None and current_reg_weight > 0:
             # Calculate raw loss
            raw_reg_loss = self.regression_loss_fn(reg_alpha, reg_beta, data.reg_y.squeeze(), data.cls_y.squeeze(), min_pKi=0, max_pKi=12, threshold=4,
                                           source_labels=data.source_y if hasattr(data, 'source_y') else None, 
                                           source_priors=self.source_priors,
                                           regul_lambda=self.regularization_lambda)
             # Apply dynamic weight
            reg_loss = raw_reg_loss * current_reg_weight
            loss_val += reg_loss

        mcls_loss = torch.tensor(0.).to(self.device) # Initialize as tensor
        if mcls_pred is not None and current_mcls_weight > 0:
            mcls_y = data.mcls_y
            # batch_size = mcls_pred.shape[0] # Get batch size from prediction tensor
            # num_classes = mcls_pred.shape[1] # Get num classes from prediction tensor
            # mcls_y = mcls_y.view(batch_size, num_classes)
            
             # Calculate raw loss
            # print('mcls_pred',mcls_pred.shape, 'mcls_y',mcls_y.shape)
            # print("mcls_y",mcls_y)
            raw_mcls_loss = self.mclassification_loss_fn(mcls_pred, mcls_y)
             # Apply dynamic weight
            mcls_loss = raw_mcls_loss * current_mcls_weight
            loss_val += mcls_loss

        # Return total loss and individual *weighted* losses
        return loss_val, reg_loss, mcls_loss


    # #### PREVIOUS LOSS FUNCTIONS ####
    # def compute_loss(self, data, reg_alpha, reg_beta, mcls_pred, pocket_loss, o_loss, cl_loss):
    #     loss_val = torch.tensor(0.).to(self.device)
    #     loss_val += pocket_loss + o_loss + cl_loss

    #     reg_loss = 0
    #     if reg_alpha is not None and reg_beta is not None:
    #         reg_loss = self.regression_loss(reg_alpha, reg_beta, data.reg_y.squeeze(), data.cls_y.squeeze(), min_pKi=0, max_pKi=12, threshold=4,
    #                                        source_labels=data.source_y if hasattr(data, 'source_y') else None, 
    #                                        source_priors=self.source_priors,
    #                                        regul_lambda=self.regularization_lambda) * self.regression_weight
    #         loss_val += reg_loss

    #     mcls_loss = 0
    #     if mcls_pred is not None:
    #         mcls_y = data.mcls_y
    #         mcls_loss = self.mclassification_loss(mcls_pred, mcls_y) * self.multiclassification_weight
    #         loss_val += mcls_loss

    #     return loss_val, reg_loss


    # def compute_prediction_loss(self, data, reg_alpha, reg_beta, mcls_pred):
    #     loss_val = torch.tensor(0.).to(self.device)

    #     reg_loss = 0
    #     if reg_alpha is not None and reg_beta is not None:
    #         reg_loss = self.regression_loss(reg_alpha, reg_beta, data.reg_y.squeeze(), data.cls_y.squeeze(), min_pKi=0, max_pKi=12, threshold=4,
    #                                        source_labels=data.source_y if hasattr(data, 'source_y') else None, 
    #                                        source_priors=self.source_priors,
    #                                        regul_lambda=self.regularization_lambda) * self.regression_weight
    #         loss_val += reg_loss

    #     mcls_loss = 0
    #     if mcls_pred is not None:
    #         mcls_y = data.mcls_y
    #         mcls_loss = self.mclassification_loss(mcls_pred, mcls_y) * self.multiclassification_weight
    #         loss_val += mcls_loss

    #     return loss_val
    


class BatchDataManager:
    def __init__(self, device, 
                 train_loader,
                 train_mutation_sampler,
                 mutation_reset_frequency=2000):
        self.device = device
        self.mutation_reset_frequency = mutation_reset_frequency
        self.mutation_access_count = 0
        self.train_iter = None
        self.train_mutation_iter = None
        self.previous_wild_only = None
        
        self.train_loader = train_loader
        self.train_iter = iter(train_loader)

        self.train_mutation_sampler = train_mutation_sampler
        self.train_mutation_loader = None
    
    def sample_data_with_retry(self, data_iter, data_loader, max_attempts=10):
        for attempt in range(max_attempts):
            try:
                data = next(data_iter)
                if data is not None:
                    return data.to(self.device), data_iter
            except StopIteration:
                data_iter = iter(data_loader)
                data = next(data_iter)
                if data is not None:
                    return data.to(self.device), data_iter
        
        raise RuntimeError(f"Failed to sample non-None data after {max_attempts} attempts.")

    def get_batch_data(self, iter_num, normal_probability, mutation_probability, wild_only=False):
        
        # only for the first time, others should be true/false
        if self.previous_wild_only is None:
            self.previous_wild_only = wild_only

        if random.random() < mutation_probability and self.train_mutation_sampler is not None:
            # Increment mutation access count
            self.mutation_access_count += 1
            
            # Reset mutation loader if access count reaches the reset frequency OR we are changing training pattern
            if (self.mutation_access_count >= self.mutation_reset_frequency) or (self.previous_wild_only != wild_only) or (self.train_mutation_loader is None):
                print(f"Resetting train_mutation_loader at iteration {iter_num}")
                gc.collect()
                torch.cuda.empty_cache()

                self.train_mutation_loader = self.train_mutation_sampler(wild_only)
                self.train_mutation_iter = None
                self.mutation_access_count = 0
            
            # Initialize or reset mutation iterator if needed
            if self.train_mutation_iter is None:
                self.train_mutation_iter = iter(self.train_mutation_loader)
            
            # Get mutation data
            data_mut, self.train_mutation_iter = self.sample_data_with_retry(self.train_mutation_iter, self.train_mutation_loader)
        else:
            data_mut = None

        self.previous_wild_only = wild_only

        data = None

        if data_mut is None or random.random() <= normal_probability:
            data, self.train_iter = self.sample_data_with_retry(self.train_iter, self.train_loader)

        return data, data_mut