// Provenance: RDKit Release_2024_03_3 TautomerEnumerator::enumerate
// (Tautomer.cpp SHA from tag 5d3c7f1df98829bb0f909f83037f508741f7a404)
// plus 9134-style cheap copies that keep SMILES map keys / visit order.
// Scoring matches 2024.03.3 scoreTautomer numerically: one symmetrized SSSR
// per input for scoreRings, loop counts for the simple SMARTS, linked
// scoreSubstructs matchers for the remaining terms, linked scoreHeteroHs.

#include <pybind11/pybind11.h>

#include <GraphMol/GraphMol.h>
#include <GraphMol/Rings.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/MolPickler.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include <GraphMol/Substruct/SubstructMatch.h>
#include <GraphMol/MolStandardize/TautomerCatalog/TautomerCatalogParams.h>
#include <GraphMol/MolStandardize/Charge.h>
#include <GraphMol/ChemTransforms/ChemTransforms.h>
#include <GraphMol/Descriptors/MolDescriptors.h>
#include <RDGeneral/types.h>

#include <algorithm>
#include <boost/dynamic_bitset.hpp>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;
using namespace RDKit;

namespace RDKit {
namespace MolStandardize {
namespace TautomerScoringFunctions {
int scoreHeteroHs(const ROMol &mol);
}  // namespace TautomerScoringFunctions
}  // namespace MolStandardize
}  // namespace RDKit

namespace {

enum class Status {
  Completed = 0,
  MaxTautomersReached,
  MaxTransformsReached,
  Canceled
};

struct Tautomer {
  ROMOL_SPTR tautomer;
  ROMOL_SPTR kekulized;
  std::string kekuleFp;
  size_t d_numModifiedAtoms = 0;
  size_t d_numModifiedBonds = 0;
  bool d_done = false;
  unsigned kekuleAtoms = 0;
  uint32_t kekuleBits = 0;

  const ROMol &getKekulized();
};

struct EnumState {
  boost::dynamic_bitset<> modifiedAtoms;
  boost::dynamic_bitset<> modifiedBonds;
};

const std::vector<MolStandardize::TautomerTransform> &transforms() {
  static MolStandardize::TautomerCatalogParams params("");
  return params.getTransforms();
}

bool setTautomerStereoAndIsoHs(const ROMol &mol, ROMol &taut, EnumState &res,
                               bool removeSp3Stereo, bool removeBondStereo,
                               bool removeIsotopicHs, bool reassignStereo) {
  bool modified = false;
  for (auto atom : mol.atoms()) {
    auto atomIdx = atom->getIdx();
    if (!res.modifiedAtoms.test(atomIdx)) {
      continue;
    }
    auto tautAtom = taut.getAtomWithIdx(atomIdx);
    if (tautAtom->getHybridization() == Atom::SP2 || removeSp3Stereo) {
      modified |= (tautAtom->getChiralTag() != Atom::CHI_UNSPECIFIED);
      tautAtom->setChiralTag(Atom::CHI_UNSPECIFIED);
      if (tautAtom->hasProp(common_properties::_CIPCode)) {
        tautAtom->clearProp(common_properties::_CIPCode);
      }
    } else {
      modified |= (tautAtom->getChiralTag() != atom->getChiralTag());
      tautAtom->setChiralTag(atom->getChiralTag());
      if (atom->hasProp(common_properties::_CIPCode)) {
        tautAtom->setProp(
            common_properties::_CIPCode,
            atom->getProp<std::string>(common_properties::_CIPCode));
      }
    }
    if (tautAtom->hasProp(common_properties::_isotopicHs) &&
        (removeIsotopicHs || !tautAtom->getTotalNumHs())) {
      tautAtom->clearProp(common_properties::_isotopicHs);
    }
  }
  for (auto bond : mol.bonds()) {
    auto bondIdx = bond->getIdx();
    if (!res.modifiedBonds.test(bondIdx)) {
      continue;
    }
    std::vector<unsigned int> bondsToClearDirs;
    if (bond->getBondType() == Bond::DOUBLE &&
        bond->getStereo() > Bond::STEREOANY) {
      for (auto atom : {bond->getBeginAtom(), bond->getEndAtom()}) {
        for (const auto &nbri :
             boost::make_iterator_range(mol.getAtomBonds(atom))) {
          const auto &obnd = mol[nbri];
          if (obnd->getBondDir() == Bond::ENDDOWNRIGHT ||
              obnd->getBondDir() == Bond::ENDUPRIGHT) {
            bondsToClearDirs.push_back(obnd->getIdx());
          }
        }
      }
    }
    auto tautBond = taut.getBondWithIdx(bondIdx);
    if (tautBond->getBondType() != Bond::DOUBLE || removeBondStereo) {
      modified |= (tautBond->getStereo() != Bond::STEREONONE);
      tautBond->setStereo(Bond::STEREONONE);
      tautBond->getStereoAtoms().clear();
      for (auto bi : bondsToClearDirs) {
        taut.getBondWithIdx(bi)->setBondDir(Bond::NONE);
      }
    } else {
      const INT_VECT &sa = bond->getStereoAtoms();
      modified |= (tautBond->getStereo() != bond->getStereo() ||
                   sa.size() != tautBond->getStereoAtoms().size());
      if (sa.size() == 2) {
        tautBond->setStereoAtoms(sa.front(), sa.back());
      }
      tautBond->setStereo(bond->getStereo());
      for (auto bi : bondsToClearDirs) {
        taut.getBondWithIdx(bi)->setBondDir(
            mol.getBondWithIdx(bi)->getBondDir());
      }
    }
  }
  if (reassignStereo) {
    static const bool cleanIt = true;
    static const bool force = true;
    MolOps::assignStereochemistry(taut, cleanIt, force);
  } else {
    taut.setProp(common_properties::_StereochemDone, 1);
  }
  return modified;
}

// Index-order kekulé identity for tautomer products of one input molecule.
// Tautomer copies keep atom/bond indices; equal H/charge/bond-type vectors
// are the same pre-sanitize structure, so sanitize+SMILES would match.
// Presence bits for defaultTautomerTransforms (37 SMARTS) are filled in the
// same atom/bond walks. Skip VF2 only when a required feature is absent.
enum KekuleFeat : uint32_t {
  HAS_P = 1u << 0,
  HAS_TRIPLE = 1u << 1,
  HAS_NPLUS = 1u << 2,
  HAS_CMINUS = 1u << 3,
  HAS_CHALC_H = 1u << 4,
  HAS_NH = 1u << 5,
  HAS_OH = 1u << 6,
  HAS_X1_CHALC_DBL = 1u << 7,
  HAS_AROM_N = 1u << 8,
  HAS_N = 1u << 9,
  HAS_S_SE_TE = 1u << 10,
  HAS_CUMUL = 1u << 11,
  HAS_R5 = 1u << 12,
  HAS_R6 = 1u << 13,
  HAS_NX3H = 1u << 14,
  HAS_HET_H = 1u << 15,
  HAS_OH_OR_NH = 1u << 16,
  HAS_KETO13_F = 1u << 17,
  HAS_KETO13_R = 1u << 18
};

void fillKekuleKeyAndBits(const ROMol &mol, std::string &key, unsigned &nAtoms,
                          uint32_t &bits) {
  nAtoms = mol.getNumAtoms();
  bits = 0;
  key.clear();
  key.reserve(nAtoms * 6 + mol.getNumBonds());
  std::vector<unsigned char> dblCount(nAtoms, 0);
  for (auto atom : mol.atoms()) {
    const int n = atom->getAtomicNum();
    const unsigned hs = atom->getTotalNumHs();
    key.push_back(static_cast<char>(n));
    key.push_back(static_cast<char>(atom->getFormalCharge() + 8));
    key.push_back(static_cast<char>(atom->getIsotope()));
    key.push_back(static_cast<char>(atom->getNumExplicitHs()));
    key.push_back(static_cast<char>(atom->getNoImplicit()));
    key.push_back(static_cast<char>(atom->getNumRadicalElectrons()));
    if (n == 15) {
      bits |= HAS_P;
    }
    if (n == 7) {
      bits |= HAS_N;
      if (atom->getFormalCharge() > 0) {
        bits |= HAS_NPLUS;
      }
      if (hs) {
        bits |= HAS_NH | HAS_HET_H | HAS_OH_OR_NH;
        if (atom->getDegree() <= 3) {
          bits |= HAS_NX3H;
        }
      }
      if (atom->getIsAromatic()) {
        bits |= HAS_AROM_N;
      }
    }
    if (n == 6 && atom->getFormalCharge() < 0) {
      bits |= HAS_CMINUS;
    }
    if (n == 8 || n == 16 || n == 34 || n == 52) {
      if (hs) {
        bits |= HAS_CHALC_H | HAS_HET_H;
        if (n == 8) {
          bits |= HAS_OH | HAS_OH_OR_NH;
        }
        for (const auto &nbri :
             boost::make_iterator_range(mol.getAtomBonds(atom))) {
          const Bond *bond = mol[nbri];
          if (bond->getBondType() != Bond::SINGLE) {
            continue;
          }
          const Atom *carbon = bond->getOtherAtom(atom);
          if (carbon->getAtomicNum() != 6) {
            continue;
          }
          for (const auto &nbri2 :
               boost::make_iterator_range(mol.getAtomBonds(carbon))) {
            const Bond *b2 = mol[nbri2];
            if (b2 == bond || b2->getBondType() != Bond::DOUBLE) {
              continue;
            }
            if (b2->getOtherAtom(carbon)->getAtomicNum() == 6) {
              bits |= HAS_KETO13_R;
              break;
            }
          }
        }
      }
      if (atom->getDegree() == 1) {
        for (const auto &nbri :
             boost::make_iterator_range(mol.getAtomBonds(atom))) {
          const Bond *bond = mol[nbri];
          if (bond->getBondType() != Bond::DOUBLE) {
            continue;
          }
          bits |= HAS_X1_CHALC_DBL;
          const Atom *carbon = bond->getOtherAtom(atom);
          if (carbon->getAtomicNum() == 6) {
            for (const auto &nbri2 :
                 boost::make_iterator_range(mol.getAtomBonds(carbon))) {
              const Bond *b2 = mol[nbri2];
              if (b2 == bond) {
                continue;
              }
              const Atom *nbr = b2->getOtherAtom(carbon);
              if (nbr->getAtomicNum() == 6 && nbr->getTotalNumHs() &&
                  nbr->getTotalDegree() == 4) {
                bits |= HAS_KETO13_F;
                break;
              }
            }
          }
        }
      }
    }
    if (n == 16 || n == 34 || n == 52) {
      bits |= HAS_S_SE_TE;
    }
  }
  for (auto bond : mol.bonds()) {
    const auto bt = bond->getBondType();
    key.push_back(static_cast<char>(bt));
    if (bt == Bond::TRIPLE) {
      bits |= HAS_TRIPLE;
    } else if (bt == Bond::DOUBLE) {
      dblCount[bond->getBeginAtomIdx()] =
          static_cast<unsigned char>(dblCount[bond->getBeginAtomIdx()] + 1);
      dblCount[bond->getEndAtomIdx()] =
          static_cast<unsigned char>(dblCount[bond->getEndAtomIdx()] + 1);
    }
  }
  for (unsigned i = 0; i < nAtoms; ++i) {
    if (dblCount[i] >= 2) {
      bits |= HAS_CUMUL;
    }
  }
  const RingInfo *ri = mol.getRingInfo();
  if (ri && ri->isInitialized()) {
    for (const auto &ring : ri->atomRings()) {
      if (ring.size() == 5) {
        bits |= HAS_R5;
      } else if (ring.size() == 6) {
        bits |= HAS_R6;
      }
    }
  }
}

const ROMol &Tautomer::getKekulized() {
  if (!kekulized) {
    RWMOL_SPTR rw(new RWMol(*tautomer, true));
    MolOps::Kekulize(*rw, false);
    kekulized = rw;
    fillKekuleKeyAndBits(*kekulized, kekuleFp, kekuleAtoms, kekuleBits);
  }
  return *kekulized;
}

bool transformPossible(size_t idx, unsigned nAtoms, uint32_t bits) {
  switch (idx) {
    case 0:
      return bits & HAS_KETO13_F;
    case 1:
      return bits & HAS_KETO13_R;
    case 2:
      return nAtoms >= 5 && (bits & HAS_X1_CHALC_DBL);
    case 3:
      return nAtoms >= 5 && (bits & HAS_CHALC_H);
    case 4:
      return bits & HAS_N;
    case 5:
      return bits & HAS_NX3H;
    case 6:
      return bits & HAS_NH;
    case 7:
      return bits & HAS_AROM_N;
    case 8:
      return (bits & HAS_AROM_N) && (bits & HAS_R6);
    case 9:
      return bits & HAS_NH;
    case 10:
      return bits & HAS_OH_OR_NH;
    case 11:
      return bits & HAS_HET_H;
    case 12:
    case 13:
    case 14:
      return nAtoms >= 5 && (bits & HAS_HET_H);
    case 15:
      return nAtoms >= 7 && (bits & HAS_HET_H);
    case 16:
      return nAtoms >= 7;
    case 17:
      return nAtoms >= 9 && (bits & HAS_OH_OR_NH);
    case 18:
      return nAtoms >= 11 && (bits & HAS_OH_OR_NH);
    case 19:
      return (bits & HAS_R5) && (bits & HAS_HET_H);
    case 20:
      return bits & HAS_R5;
    case 21:
      return (bits & HAS_CUMUL) && (bits & HAS_X1_CHALC_DBL);
    case 22:
      return (bits & HAS_CHALC_H) && (bits & HAS_TRIPLE);
    case 23:
      return bits & HAS_NPLUS;
    case 24:
      return (bits & HAS_NPLUS) && (bits & HAS_OH);
    case 25:
      return (bits & HAS_OH) && (bits & HAS_N);
    case 26:
      return (bits & HAS_N) && (bits & HAS_X1_CHALC_DBL);
    case 27:
      return nAtoms >= 7 && (bits & HAS_OH) && (bits & HAS_N);
    case 28:
      return (bits & HAS_OH) && (bits & HAS_N);
    case 29:
      return (bits & HAS_OH) && (bits & HAS_TRIPLE);
    case 30:
      return (bits & HAS_NH) && (bits & HAS_CUMUL);
    case 31:
      return bits & HAS_S_SE_TE;
    case 32:
      return (bits & HAS_S_SE_TE) && (bits & HAS_OH);
    case 33:
      return bits & HAS_TRIPLE;
    case 34:
      return (bits & HAS_NPLUS) && (bits & HAS_CMINUS);
    case 35:
    case 36:
      return bits & HAS_P;
    default:
      return true;
  }
}

// Stereo-free resonance identity: parent aromatic bonds collapse kekulé
// pairs; atoms use total H. Do not skip from this key when the input has
// stereo. Isomeric SMILES keys can still differ after stereo stripping.
std::string aromaticInvariant(ROMol &product, const ROMol &parent) {
  product.updatePropertyCache(false);
  std::string key;
  key.reserve(product.getNumAtoms() * 5 + product.getNumBonds());
  for (auto atom : product.atoms()) {
    key.push_back(static_cast<char>(atom->getAtomicNum()));
    key.push_back(static_cast<char>(atom->getFormalCharge() + 8));
    key.push_back(static_cast<char>(atom->getIsotope()));
    key.push_back(static_cast<char>(atom->getTotalNumHs()));
    key.push_back(static_cast<char>(atom->getNumRadicalElectrons()));
  }
  for (auto bond : product.bonds()) {
    const auto parentBond = parent.getBondWithIdx(bond->getIdx());
    if (parentBond->getIsAromatic()) {
      key.push_back('A');
    } else {
      key.push_back(static_cast<char>(bond->getBondType()));
    }
  }
  return key;
}

// Same byte layout as aromaticInvariant after the product copy+H/bond/charge
// patches, without allocating RWMol. Tautomer H moves only change total H on
// the match endpoints (then NoImplicit); other atoms keep kmol total H.
std::string aromaticInvariantFromKekule(
    const ROMol &kmol, const ROMol &parent, const MatchVectType &match,
    const MolStandardize::TautomerTransform &transform) {
  const int firstIdx = match.front().second;
  const int lastIdx = match.back().second;
  std::string key;
  key.reserve(kmol.getNumAtoms() * 5 + kmol.getNumBonds());
  for (auto atom : kmol.atoms()) {
    const int idx = static_cast<int>(atom->getIdx());
    int charge = atom->getFormalCharge();
    if (!transform.Charges.empty()) {
      unsigned int ci = 0;
      for (const auto &pair : match) {
        if (static_cast<int>(pair.second) == idx) {
          charge += transform.Charges[ci];
        }
        ++ci;
      }
    }
    unsigned int totalHs = atom->getTotalNumHs();
    if (idx == firstIdx) {
      totalHs = static_cast<unsigned int>(
          std::max(0, static_cast<int>(atom->getTotalNumHs()) - 1));
    }
    if (idx == lastIdx) {
      if (idx == firstIdx) {
        totalHs += 1;
      } else {
        totalHs = atom->getTotalNumHs() + 1;
      }
    }
    key.push_back(static_cast<char>(atom->getAtomicNum()));
    key.push_back(static_cast<char>(charge + 8));
    key.push_back(static_cast<char>(atom->getIsotope()));
    key.push_back(static_cast<char>(totalHs));
    key.push_back(static_cast<char>(atom->getNumRadicalElectrons()));
  }
  unsigned int overlayIdx[16];
  Bond::BondType overlayType[16];
  unsigned int nOverlay = 0;
  unsigned int bi = 0;
  const unsigned int nTB = transform.Mol->getNumBonds();
  if (nTB > 16) {
    throw std::runtime_error("tautomer transform has too many bonds");
  }
  for (unsigned int i = 0; i < nTB; ++i) {
    const auto tbond = transform.Mol->getBondWithIdx(i);
    const Bond *bond = kmol.getBondBetweenAtoms(
        match[tbond->getBeginAtomIdx()].second,
        match[tbond->getEndAtomIdx()].second);
    if (!bond) {
      throw std::runtime_error("required tautomer bond not found");
    }
    Bond::BondType bondtype = bond->getBondType();
    if (!transform.BondTypes.empty()) {
      bondtype = transform.BondTypes[bi++];
    } else if (bondtype == Bond::SINGLE) {
      bondtype = Bond::DOUBLE;
    } else if (bondtype == Bond::DOUBLE) {
      bondtype = Bond::SINGLE;
    }
    overlayIdx[nOverlay] = bond->getIdx();
    overlayType[nOverlay] = bondtype;
    ++nOverlay;
  }
  for (auto bond : kmol.bonds()) {
    const auto parentBond = parent.getBondWithIdx(bond->getIdx());
    if (parentBond->getIsAromatic()) {
      key.push_back('A');
      continue;
    }
    Bond::BondType bt = bond->getBondType();
    const unsigned int bidx = bond->getIdx();
    for (unsigned int i = 0; i < nOverlay; ++i) {
      if (overlayIdx[i] == bidx) {
        bt = overlayType[i];
        break;
      }
    }
    key.push_back(static_cast<char>(bt));
  }
  return key;
}

bool isHeteroAtom(const Atom *atom) {
  const int n = atom->getAtomicNum();
  return n != 1 && n != 6;
}

int scoreRingsFromBondRings(const ROMol &mol, const VECT_INT_VECT &bondRings) {
  int score = 0;
  for (const auto &bring : bondRings) {
    bool allC = true;
    bool allAromatic = true;
    for (const auto bidx : bring) {
      const Bond *bnd = mol.getBondWithIdx(bidx);
      if (!bnd->getIsAromatic()) {
        allAromatic = false;
        break;
      }
      if (bnd->getBeginAtom()->getAtomicNum() != 6 ||
          bnd->getEndAtom()->getAtomicNum() != 6) {
        allC = false;
      }
    }
    if (allAromatic) {
      score += 100;
      if (allC) {
        score += 150;
      }
    }
  }
  return score;
}

int scoreSimpleSubstructs(const ROMol &mol) {
  int score = 0;
  const RingInfo *ri = mol.getRingInfo();
  for (const Bond *bond : mol.bonds()) {
    const Atom *a = bond->getBeginAtom();
    const Atom *b = bond->getEndAtom();
    const Bond::BondType bt = bond->getBondType();
    const bool dbl = bt == Bond::DOUBLE;
    const bool dblOrArom = dbl || bond->getIsAromatic() || bt == Bond::AROMATIC;
    if (dblOrArom) {
      const int an = a->getAtomicNum();
      const int bn = b->getAtomicNum();
      if (an == 8 || bn == 8) {
        const int other = an == 8 ? bn : an;
        if (other == 6) {
          score += 2;
        } else if (other == 7) {
          score += 2;
        } else if (other == 15) {
          score += 2;
        }
      }
      const bool exocyclic =
          !(ri && ri->isInitialized() && ri->numBondRings(bond->getIdx()));
      if (exocyclic) {
        if (a->getIsAromatic() && an == 6 && bn == 7) {
          score -= 1;
        } else if (b->getIsAromatic() && bn == 6 && an == 7) {
          score -= 1;
        }
      }
    }
    if (dbl) {
      if (a->getAtomicNum() == 6 && !a->getIsAromatic() && isHeteroAtom(b)) {
        score += 1;
      } else if (b->getAtomicNum() == 6 && !b->getIsAromatic() &&
                 isHeteroAtom(a)) {
        score += 1;
      }
    }
  }
  for (const Atom *atom : mol.atoms()) {
    if (atom->getAtomicNum() == 6 && !atom->getIsAromatic() &&
        atom->getTotalDegree() == 4 && atom->getTotalNumHs() == 3) {
      score += 1;
    }
    if (atom->getAtomicNum() != 7) {
      continue;
    }
    bool hasCDbl = false;
    bool hasOH = false;
    for (const auto &nbri :
         boost::make_iterator_range(mol.getAtomBonds(atom))) {
      const Bond *bond = mol[nbri];
      const Atom *nbr = bond->getOtherAtom(atom);
      if (nbr->getAtomicNum() == 6 && bond->getBondType() == Bond::DOUBLE) {
        hasCDbl = true;
      }
      if (nbr->getAtomicNum() == 8 && nbr->getTotalNumHs() >= 1 &&
          bond->getBondType() == Bond::SINGLE) {
        hasOH = true;
      }
    }
    if (hasCDbl && hasOH) {
      score += 4;
    }
  }
  return score;
}

const std::vector<std::pair<ROMOL_SPTR, int>> &hardSubstructTerms() {
  static const std::vector<std::pair<ROMOL_SPTR, int>> terms = [] {
    std::vector<std::pair<ROMOL_SPTR, int>> v;
    auto add = [&](const char *smarts, int score) {
      v.emplace_back(ROMOL_SPTR(SmartsToMol(smarts)), score);
    };
    add("[#6]1([#6]=[#6][#6]([#6]=[#6]1)=,:[N,S,O])=,:[N,S,O]", 25);
    add("[C](=[!#1;!#6])[!#1;!#6]", 2);
    add("[#7]C(=[NR0])[#7H0]", 1);
    add("[#7;R][#6;R]([N])=[#7;R]", 2);
    add("[#6]=[N+]([O-])[OH]", -4);
    return v;
  }();
  return terms;
}

int scoreHardSubstructs(const ROMol &mol) {
  int score = 0;
  SubstructMatchParameters params;
  for (const auto &term : hardSubstructTerms()) {
    if (!term.first) {
      continue;
    }
    const auto matches = SubstructMatch(mol, *term.first, params);
    score += static_cast<int>(matches.size()) * term.second;
  }
  return score;
}

int scoreTautomerLocal(const ROMol &mol, const VECT_INT_VECT &bondRings) {
  return scoreRingsFromBondRings(mol, bondRings) + scoreSimpleSubstructs(mol) +
         scoreHardSubstructs(mol) +
         MolStandardize::TautomerScoringFunctions::scoreHeteroHs(mol);
}

bool molHasStereo(const ROMol &mol) {
  for (auto atom : mol.atoms()) {
    if (atom->getChiralTag() != Atom::CHI_UNSPECIFIED) {
      return true;
    }
  }
  for (auto bond : mol.bonds()) {
    if (bond->getStereo() > Bond::STEREONONE) {
      return true;
    }
  }
  return false;
}

std::string patchedKekuleKey(const std::string &parent, const ROMol &kmol,
                             const MatchVectType &match,
                             const MolStandardize::TautomerTransform &transform,
                             EnumState &res) {
  std::string fp = parent;
  const int firstIdx = match.front().second;
  const int lastIdx = match.back().second;
  const Atom *first = kmol.getAtomWithIdx(firstIdx);
  const Atom *last = kmol.getAtomWithIdx(lastIdx);
  res.modifiedAtoms.set(firstIdx);
  res.modifiedAtoms.set(lastIdx);
  fp[static_cast<size_t>(firstIdx) * 6 + 3] = static_cast<char>(
      std::max(0, static_cast<int>(first->getTotalNumHs()) - 1));
  fp[static_cast<size_t>(firstIdx) * 6 + 4] = 1;
  fp[static_cast<size_t>(lastIdx) * 6 + 3] =
      static_cast<char>(last->getTotalNumHs() + 1);
  fp[static_cast<size_t>(lastIdx) * 6 + 4] = 1;
  const size_t bondBase = static_cast<size_t>(kmol.getNumAtoms()) * 6;
  unsigned int bi = 0;
  for (size_t i = 0; i < transform.Mol->getNumBonds(); ++i) {
    const auto tbond = transform.Mol->getBondWithIdx(i);
    const Bond *bond = kmol.getBondBetweenAtoms(
        match[tbond->getBeginAtomIdx()].second,
        match[tbond->getEndAtomIdx()].second);
    if (!bond) {
      throw std::runtime_error("required tautomer bond not found");
    }
    Bond::BondType bondtype = bond->getBondType();
    if (!transform.BondTypes.empty()) {
      bondtype = transform.BondTypes[bi];
      ++bi;
    } else {
      if (bondtype == Bond::SINGLE) {
        bondtype = Bond::DOUBLE;
      } else if (bondtype == Bond::DOUBLE) {
        bondtype = Bond::SINGLE;
      }
    }
    fp[bondBase + bond->getIdx()] = static_cast<char>(bondtype);
    res.modifiedBonds.set(bond->getIdx());
  }
  if (!transform.Charges.empty()) {
    unsigned int ci = 0;
    for (const auto &pair : match) {
      const Atom *atom = kmol.getAtomWithIdx(pair.second);
      fp[static_cast<size_t>(pair.second) * 6 + 1] = static_cast<char>(
          atom->getFormalCharge() + transform.Charges[ci++] + 8);
    }
  }
  return fp;
}

bool sanitizeProduct(RWMol &product, const RingInfo &cachedRings) {
  try {
    product.clearComputedProps();
    *product.getRingInfo() = cachedRings;
    product.updatePropertyCache(false);
    MolOps::Kekulize(product);
    MolOps::setAromaticity(product);
    MolOps::adjustHs(product);
    return true;
  } catch (const KekulizeException &) {
    return false;
  }
}

std::pair<ROMOL_SPTR, Status> enumerateAndPick(
    const ROMol &mol, unsigned int maxTautomers, bool removeSp3Stereo,
    bool removeBondStereo) {
  const auto &tx = transforms();
  const unsigned int maxTransforms = 100;
  const bool removeIsotopicHs = true;
  const bool reassignStereo = true;

  std::map<std::string, Tautomer> tautomers;
  std::unordered_set<std::string> seenKekule;
  std::unordered_set<std::string> seenInvar;
  EnumState res;
  const bool parentHasStereo = molHasStereo(mol);
  std::string smi = MolToSmiles(mol, true);
  ROMOL_SPTR taut(new ROMol(mol));
  if (taut->needsUpdatePropertyCache()) {
    taut->updatePropertyCache(false);
  }
  if (!taut->getRingInfo()->isSymmSssr()) {
    MolOps::symmetrizeSSSR(*taut);
  }
  RWMol ringProbe(*taut);
  ringProbe.clearComputedProps();
  MolOps::findSSSR(ringProbe);
  const RingInfo cachedRings = *ringProbe.getRingInfo();
  VECT_INT_VECT scoreBondRings;
  {
    RWMol ringSym(mol);
    if (ringSym.needsUpdatePropertyCache()) {
      ringSym.updatePropertyCache(false);
    }
    MolOps::symmetrizeSSSR(ringSym);
    scoreBondRings = ringSym.getRingInfo()->bondRings();
  }
  tautomers.emplace(smi, Tautomer{taut, nullptr, std::string(), 0, 0, false});
  (void)tautomers.begin()->second.getKekulized();
  seenKekule.insert(tautomers.begin()->second.kekuleFp);
  if (!parentHasStereo) {
    ROMol seed(mol);
    seenInvar.insert(aromaticInvariant(seed, mol));
  }

  res.modifiedAtoms.resize(mol.getNumAtoms());
  res.modifiedBonds.resize(mol.getNumBonds());

  bool completed = false;
  bool bailOut = false;
  unsigned int nTransforms = 0;
  Status status = Status::Completed;

  while (!completed && !bailOut) {
    for (auto &smilesTautomerPair : tautomers) {
      if (smilesTautomerPair.second.d_done) {
        continue;
      }
      for (size_t ti = 0; ti < tx.size(); ++ti) {
        if (bailOut) {
          break;
        }
        const auto &transform = tx[ti];
        const auto &kmol = smilesTautomerPair.second.getKekulized();
        if (!transformPossible(ti, smilesTautomerPair.second.kekuleAtoms,
                               smilesTautomerPair.second.kekuleBits)) {
          continue;
        }
        std::vector<MatchVectType> matches;
        unsigned int matched = SubstructMatch(kmol, *(transform.Mol), matches);
        if (!matched) {
          continue;
        }
        ++nTransforms;
        for (const auto &match : matches) {
          if (nTransforms >= maxTransforms) {
            status = Status::MaxTransformsReached;
            bailOut = true;
          } else if (tautomers.size() >= maxTautomers) {
            status = Status::MaxTautomersReached;
            bailOut = true;
          }
          if (bailOut) {
            break;
          }
          std::string fp = patchedKekuleKey(
              smilesTautomerPair.second.kekuleFp, kmol, match, transform, res);
          if (!seenKekule.insert(fp).second) {
            continue;
          }
          // Stereo-free: H15 resonance skip before the product copy. seenInvar
          // already holds aromaticInvariant of accepted products; a matching
          // pre-copy key is the same skip without new RWMol(kmol).
          if (!parentHasStereo &&
              !seenInvar
                   .insert(aromaticInvariantFromKekule(kmol, mol, match,
                                                       transform))
                   .second) {
            continue;
          }
          RWMOL_SPTR product(new RWMol(kmol, true));
          int firstIdx = match.front().second;
          int lastIdx = match.back().second;
          Atom *first = product->getAtomWithIdx(firstIdx);
          Atom *last = product->getAtomWithIdx(lastIdx);
          first->setNumExplicitHs(
              std::max(0, static_cast<int>(first->getTotalNumHs()) - 1));
          last->setNumExplicitHs(last->getTotalNumHs() + 1);
          first->setNoImplicit(true);
          last->setNoImplicit(true);
          unsigned int bi = 0;
          for (size_t i = 0; i < transform.Mol->getNumBonds(); ++i) {
            const auto tbond = transform.Mol->getBondWithIdx(i);
            Bond *bond = product->getBondBetweenAtoms(
                match[tbond->getBeginAtomIdx()].second,
                match[tbond->getEndAtomIdx()].second);
            if (!bond) {
              throw std::runtime_error("required tautomer bond not found");
            }
            if (!transform.BondTypes.empty()) {
              bond->setBondType(transform.BondTypes[bi]);
              ++bi;
            } else {
              Bond::BondType bondtype = bond->getBondType();
              if (bondtype == Bond::SINGLE) {
                bond->setBondType(Bond::DOUBLE);
              }
              if (bondtype == Bond::DOUBLE) {
                bond->setBondType(Bond::SINGLE);
              }
            }
          }
          if (!transform.Charges.empty()) {
            unsigned int ci = 0;
            for (const auto &pair : match) {
              Atom *atom = product->getAtomWithIdx(pair.second);
              atom->setFormalCharge(atom->getFormalCharge() +
                                    transform.Charges[ci++]);
            }
          }
          if (!sanitizeProduct(*product, cachedRings)) {
            continue;
          }
          setTautomerStereoAndIsoHs(mol, *product, res, removeSp3Stereo,
                                    removeBondStereo, removeIsotopicHs,
                                    reassignStereo);
          std::string tsmiles = MolToSmiles(*product, true);
          if (tautomers.find(tsmiles) != tautomers.end()) {
            continue;
          }
          for (size_t i = 0; i < mol.getNumBonds(); i++) {
            auto molBondType = mol.getBondWithIdx(i)->getBondType();
            auto tautBondType = product->getBondWithIdx(i)->getBondType();
            if (molBondType != tautBondType && !res.modifiedBonds.test(i)) {
              res.modifiedBonds.set(i);
            }
          }
          tautomers[tsmiles] = Tautomer{std::move(product), nullptr, std::string(),
                                        res.modifiedAtoms.count(),
                                        res.modifiedBonds.count(), false};
        }
      }
      smilesTautomerPair.second.d_done = true;
    }
    completed = true;
    size_t maxNumModifiedAtoms = res.modifiedAtoms.count();
    size_t maxNumModifiedBonds = res.modifiedBonds.count();
    for (auto it = tautomers.begin(); it != tautomers.end();) {
      auto &item = it->second;
      if (!item.d_done) {
        completed = false;
      }
      if ((item.d_numModifiedAtoms < maxNumModifiedAtoms ||
           item.d_numModifiedBonds < maxNumModifiedBonds) &&
          setTautomerStereoAndIsoHs(mol, *item.tautomer, res, removeSp3Stereo,
                                    removeBondStereo, removeIsotopicHs,
                                    reassignStereo)) {
        Tautomer stored = std::move(item);
        it = tautomers.erase(it);
        stored.d_numModifiedAtoms = maxNumModifiedAtoms;
        stored.d_numModifiedBonds = maxNumModifiedBonds;
        auto insertRes = tautomers.insert(std::make_pair(
            MolToSmiles(*stored.tautomer), std::move(stored)));
        if (insertRes.second) {
          it = insertRes.first;
        }
      } else {
        ++it;
      }
    }
    if (bailOut && tautomers.size() < maxTautomers &&
        status == Status::MaxTautomersReached) {
      status = Status::Completed;
      bailOut = false;
    }
  }

  ROMOL_SPTR bestMol;
  if (tautomers.size() == 1) {
    bestMol = tautomers.begin()->second.tautomer;
  } else {
    int bestScore = std::numeric_limits<int>::min();
    std::string bestSmiles;
    for (const auto &item : tautomers) {
      auto score = scoreTautomerLocal(*item.second.tautomer, scoreBondRings);
      if (score > bestScore) {
        bestScore = score;
        bestSmiles = item.first;
        bestMol = item.second.tautomer;
      } else if (score == bestScore && item.first < bestSmiles) {
        bestSmiles = item.first;
        bestMol = item.second.tautomer;
      }
    }
  }
  ROMOL_SPTR picked(new ROMol(*bestMol));
  MolOps::assignStereochemistry(*picked, true, true);
  return {picked, status};
}

bool hasSubstruct(const ROMol &mol, const ROMol &query) {
  std::vector<MatchVectType> hits;
  return SubstructMatch(mol, query, hits, true, true, false, false, 1) > 0;
}

const std::vector<ROMOL_SPTR> &chemblRestPatterns() {
  static const std::vector<ROMOL_SPTR> patterns = [] {
    const char *smarts[] = {
        "[N;X3](=[O])=[O]",
        "[N;X2]#[N;X1]",
        "[N;X4;v4;+0]",
        "[O;X2;v3;+0]",
        "[S+0;D3](=[O])",
        "[SH1+1;D3](=[O])",
        "[O]=[S;D2;+0]",
        "[C]([OH1;D1])=;!@[NH1]",
        "[C]([OH1;D1])=;!@[NH0]",
        "[F,Cl,Br,I;X0;+0]",
        "[C,N;-;D2,D3]-[N+2;D3]-[O-;D1]",
        "[N-][N+]#[N]",
        "[Li,Na,K;+0]-[#7,#8;+0]",
        "OC(=O)C(O)C(O)C(=O)O",
    };
    std::vector<ROMOL_SPTR> out;
    for (const char *s : smarts) {
      std::unique_ptr<RWMol> q(SmartsToMol(s));
      if (q) {
        out.emplace_back(q.release());
      }
    }
    return out;
  }();
  return patterns;
}

bool chemblMayChange(const ROMol &mol) {
  if (mol.getNumConformers()) {
    return true;
  }
  std::vector<int> mapping;
  if (MolOps::getMolFrags(mol, mapping) != 1) {
    return true;
  }
  for (auto atom : mol.atoms()) {
    if (atom->getIsotope() || atom->getFormalCharge()) {
      return true;
    }
  }
  for (const auto &pattern : chemblRestPatterns()) {
    if (hasSubstruct(mol, *pattern)) {
      return true;
    }
  }
  return false;
}

using SaltSize = std::pair<unsigned int, unsigned int>;

const std::map<SaltSize, std::vector<ROMOL_SPTR>> &saltPatterns() {
  static const std::map<SaltSize, std::vector<ROMOL_SPTR>> patterns = [] {
    const char *salts[] = {
        "[Na+]", "[K+]", "Cl", "C1=NC=CC=C1", "[O-][Cl+3]([O-])([O-])[O-]",
        "O=CN(C)C", "[No]", "C", "C1COCCO1", "CCCCCCN(C)C", "CCCCCC",
        "CN(C)C(=N)N", "CN(C)C",
        "c1ccc([B-](c2ccccc2)(c2ccccc2)c2ccccc2)cc1",
        "[He]", "[Li]", "[Be]", "[Ne]", "[Na]", "[Mg]", "[Al]", "[Si]", "[Ar]",
        "[K]", "[Ca]", "[Sc]", "[Ti]", "[V]", "[Cr]", "[Mn]", "[Fe]", "[Co]",
        "[Ni]", "[Cu]", "[Zn]", "[Ga]", "[Ge]", "[As]", "[Se]", "[Kr]", "[Rb]",
        "[Sr]", "[Y]", "[Zr]", "[Nb]", "[Mo]", "[Tc]", "[Ru]", "[Rh]", "[Pd]",
        "[Ag]", "[Cd]", "[In]", "[Sn]", "[Sb]", "[Te]", "[Xe]", "[Cs]", "[Ba]",
        "[La]", "[Ce]", "[Pr]", "[Nd]", "[Pm]", "[Sm]", "[Eu]", "[Gd]", "[Tb]",
        "[Dy]", "[Ho]", "[Er]", "[Tm]", "[Yb]", "[Lu]", "[Hf]", "[Ta]", "[W]",
        "[Re]", "[Os]", "[Ir]", "[Pt]", "[Au]", "[Hg]", "[Tl]", "[Pb]", "[Bi]",
        "[Po]", "[At]", "[Rn]", "[Fr]", "[Ra]", "[Ac]", "[Th]", "[Pa]", "[U]",
        "[Np]", "[Pu]", "[Am]", "[Cm]", "[Bk]", "[Cf]", "[Es]", "[Fm]", "[Md]",
        "[No]", "[Lr]", "[Rf]", "[Db]", "[Sg]", "[Bh]", "[Hs]", "[Mt]", "[Ds]",
        "[Rg]", "[Cn]", "[Nh]", "[Fl]", "[Mc]", "[Lv]", "[Ts]", "[Og]",
    };
    std::map<SaltSize, std::vector<ROMOL_SPTR>> out;
    for (const char *s : salts) {
      std::unique_ptr<RWMol> salt(SmilesToMol(s));
      if (!salt) {
        continue;
      }
      SaltSize size{salt->getNumAtoms(), salt->getNumBonds()};
      out[size].emplace_back(salt.release());
    }
    return out;
  }();
  return patterns;
}

ROMOL_SPTR removeSupplementarySalts(const ROMol &mol) {
  auto fragsRaw = MolOps::getMolFrags(mol, false);
  std::vector<ROMOL_SPTR> frags;
  frags.reserve(fragsRaw.size());
  for (const auto &frag : fragsRaw) {
    std::unique_ptr<ROMol> stripped(
        MolOps::removeHs(*frag, false, false, false));
    RWMol rw(*stripped);
    rw.updatePropertyCache(false);
    MolOps::setAromaticity(rw);
    frags.emplace_back(new ROMol(rw));
  }
  std::vector<char> keep(frags.size(), 1);
  const auto &salts = saltPatterns();
  for (size_t i = 0; i < frags.size(); ++i) {
    SaltSize size{frags[i]->getNumAtoms(), frags[i]->getNumBonds()};
    auto found = salts.find(size);
    if (found == salts.end()) {
      continue;
    }
    for (const auto &salt : found->second) {
      if (keep[i] && hasSubstruct(*frags[i], *salt) &&
          hasSubstruct(*salt, *frags[i])) {
        keep[i] = 0;
      }
      bool any = false;
      for (char k : keep) {
        if (k) {
          any = true;
          break;
        }
      }
      if (!any) {
        return ROMOL_SPTR(new ROMol(mol));
      }
    }
  }
  std::vector<ROMOL_SPTR> kept;
  for (size_t i = 0; i < frags.size(); ++i) {
    if (keep[i]) {
      kept.push_back(frags[i]);
    }
  }
  if (kept.size() == 1) {
    return kept[0];
  }
  ROMOL_SPTR res = kept[0];
  for (size_t i = 1; i < kept.size(); ++i) {
    std::unique_ptr<ROMol> combined(combineMols(*res, *kept[i]));
    res.reset(combined.release());
  }
  return res;
}

const std::vector<std::pair<ROMOL_SPTR, ROMOL_SPTR>> &neutralizingReactions() {
  static const std::vector<std::pair<ROMOL_SPTR, ROMOL_SPTR>> reactions = [] {
    const std::pair<const char *, const char *> pairs[] = {
        {"[n+;H]", "n"},
        {"[N+;!H0]", "N"},
        {"[$([O-]);!$([O-][#7])]", "O"},
        {"[S-;X1]", "S"},
        {"[$([N-;X2]S(=O)=O)]", "N"},
        {"[$([N-;X2][C,N]=C)]", "N"},
        {"[n-]", "[nH]"},
        {"[$([S-]=O)]", "S"},
        {"[$([N-]C=O)]", "N"},
    };
    std::vector<std::pair<ROMOL_SPTR, ROMOL_SPTR>> out;
    for (const auto &pair : pairs) {
      std::unique_ptr<RWMol> reactant(SmartsToMol(pair.first));
      std::unique_ptr<RWMol> product(SmilesToMol(pair.second, 0, false));
      if (reactant && product) {
        out.emplace_back(reactant.release(), product.release());
      }
    }
    return out;
  }();
  return reactions;
}

ROMOL_SPTR unchargePapyrus(const ROMol &mol) {
  bool charged = false;
  for (auto atom : mol.atoms()) {
    if (atom->getFormalCharge()) {
      charged = true;
      break;
    }
  }
  if (!charged) {
    return ROMOL_SPTR(new ROMol(mol));
  }
  MolStandardize::Uncharger uncharger;
  std::unique_ptr<ROMol> neutralized(uncharger.uncharge(mol));
  RWMol rw(*neutralized);
  for (const auto &reaction : neutralizingReactions()) {
    while (hasSubstruct(rw, *reaction.first)) {
      auto reps = replaceSubstructs(rw, *reaction.first, *reaction.second);
      if (reps.empty()) {
        break;
      }
      rw = RWMol(*reps[0]);
    }
  }
  MolOps::sanitizeMol(rw);
  return ROMOL_SPTR(new ROMol(rw));
}

struct PapyrusOut {
  std::string kind;
  std::string smiles;
  std::string error;
  std::string pickle;
};

PapyrusOut papyrusFromMol(const ROMol &mol, bool pickle_ok) {
  ROMOL_SPTR cur = removeSupplementarySalts(mol);
  std::vector<int> mapping;
  if (MolOps::getMolFrags(*cur, mapping) > 1) {
    return {"mixture", "", "Multiple fragments detected", ""};
  }
  cur = unchargePapyrus(*cur);
  double mw = Descriptors::calcExactMW(*cur);
  if (mw < 100.0 || mw > 2000.0) {
    return {"non_small", "", "Molecular weight outside allowed range", ""};
  }
  auto picked = enumerateAndPick(*cur, 0xffffffffu, true, true);
  if (chemblMayChange(*picked.first)) {
    std::string blob;
    MolPickler::pickleMol(*picked.first, blob);
    return {"need_chembl2", "", "", blob};
  }
  std::string blob;
  if (pickle_ok) {
    MolPickler::pickleMol(*picked.first, blob);
  }
  return {"ok", MolToSmiles(*picked.first, false), "", blob};
}

PapyrusOut normalizeOnceNative(const std::string &smi) {
  std::unique_ptr<RWMol> mol(SmilesToMol(smi));
  if (!mol) {
    return {"invalid", "", "Could not parse SMILES", ""};
  }
  if (chemblMayChange(*mol)) {
    return {"need_chembl1", "", "", ""};
  }
  return papyrusFromMol(*mol, false);
}

PapyrusOut papyrusRestNative(const std::string &raw) {
  ROMol mol;
  MolPickler::molFromPickle(raw, mol);
  return papyrusFromMol(mol, true);
}

py::tuple packPapyrus(const PapyrusOut &out) {
  return py::make_tuple(out.kind, out.smiles, out.error, py::bytes(out.pickle));
}

py::tuple pick_canonical_tautomer(py::bytes pickle, unsigned int max_tautomers,
                                  bool remove_sp3, bool remove_bond) {
  std::string raw = pickle;
  std::string out;
  int status = 0;
  {
    py::gil_scoped_release release;
    ROMol mol;
    MolPickler::molFromPickle(raw, mol);
    auto result =
        enumerateAndPick(mol, max_tautomers, remove_sp3, remove_bond);
    MolPickler::pickleMol(*result.first, out);
    status = static_cast<int>(result.second);
  }
  return py::make_tuple(py::bytes(out), status);
}

py::tuple normalize_once(const std::string &smi) {
  PapyrusOut out;
  try {
    py::gil_scoped_release release;
    out = normalizeOnceNative(smi);
  } catch (const std::exception &exc) {
    out = {"error", "", "Molecule could not be standardized", ""};
    (void)exc;
  }
  return packPapyrus(out);
}

py::tuple papyrus_rest(py::bytes pickle) {
  std::string raw = pickle;
  PapyrusOut out;
  try {
    py::gil_scoped_release release;
    out = papyrusRestNative(raw);
  } catch (const std::exception &exc) {
    out = {"error", "", "Molecule could not be standardized", ""};
    (void)exc;
  }
  return packPapyrus(out);
}

}  // namespace

PYBIND11_MODULE(tautomer_ext, m) {
  m.def("pick_canonical_tautomer", &pick_canonical_tautomer,
        py::arg("pickle"), py::arg("max_tautomers"), py::arg("remove_sp3"),
        py::arg("remove_bond"));
  m.def("normalize_once", &normalize_once, py::arg("smiles"));
  m.def("papyrus_rest", &papyrus_rest, py::arg("pickle"));
}
