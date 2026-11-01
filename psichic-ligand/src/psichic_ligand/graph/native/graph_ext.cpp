// Provenance: atom/bond feature maps match utils/featurizers.py from_rdmol
// (PSICHIC+ original Python, rdkit 2024.3.3). Clique merge and Kruskal MST
// are native. The dense PE matmul stays in Python. Timed smiles_to_graph
// parses SMILES in C++ so it does not pickle a Python mol.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/MolPickler.h>
#include <GraphMol/RingInfo.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;
using namespace RDKit;

namespace {

py::array_t<int64_t> vec_array(const std::vector<int64_t> &v) {
  auto arr = py::array_t<int64_t>(v.size());
  if (!v.empty()) {
    std::memcpy(arr.mutable_data(), v.data(), v.size() * sizeof(int64_t));
  }
  return arr;
}

py::array_t<int64_t> mat_array(size_t rows, size_t cols,
                               const std::vector<int64_t> &v) {
  auto arr = py::array_t<int64_t>(
      {static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
  if (!v.empty()) {
    std::memcpy(arr.mutable_data(), v.data(), v.size() * sizeof(int64_t));
  }
  return arr;
}

int map_atom_code(int atomic_num) {
  switch (atomic_num) {
    case 5:
      return 1;
    case 6:
      return 2;
    case 7:
      return 3;
    case 8:
      return 4;
    case 15:
      return 5;
    case 16:
      return 6;
    case 34:
      return 7;
    case 9:
    case 17:
    case 35:
    case 53:
      return 8;
    default:
      break;
  }
  if ((atomic_num >= 3 && atomic_num <= 4) || atomic_num == 11 ||
      atomic_num == 12 || atomic_num == 13 ||
      (atomic_num >= 19 && atomic_num <= 31) ||
      (atomic_num >= 37 && atomic_num <= 50) ||
      (atomic_num >= 55 && atomic_num <= 83) ||
      (atomic_num >= 87 && atomic_num <= 103)) {
    return 9;
  }
  return 0;
}

int chirality_index(Atom::ChiralType tag) {
  const int value = static_cast<int>(tag);
  if (value >= 0 && value <= 8) {
    return value;
  }
  return 0;
}

int degree_index(unsigned int degree) {
  if (degree <= 10) {
    return static_cast<int>(degree);
  }
  return 10;
}

int charge_index(int charge) {
  if (charge >= -5 && charge <= 9) {
    return charge + 5;
  }
  return 14;
}

int hs_index(unsigned int hs) {
  if (hs <= 8) {
    return static_cast<int>(hs);
  }
  return 8;
}

int radical_index(unsigned int rad) {
  if (rad <= 4) {
    return static_cast<int>(rad);
  }
  return 4;
}

int hybridization_index(Atom::HybridizationType hyb) {
  switch (hyb) {
    case Atom::UNSPECIFIED:
      return 0;
    case Atom::S:
      return 1;
    case Atom::SP:
      return 2;
    case Atom::SP2:
      return 3;
    case Atom::SP3:
      return 4;
    case Atom::SP2D:
      return 0;
    case Atom::SP3D:
      return 5;
    case Atom::SP3D2:
      return 6;
    case Atom::OTHER:
      return 7;
    default:
      return 0;
  }
}

int bond_type_index(Bond::BondType type) {
  const int value = static_cast<int>(type);
  if (value < 0 || value > 21) {
    throw std::runtime_error("bond type is not in the PSICHIC e_map");
  }
  return value;
}

int stereo_index(Bond::BondStereo stereo) {
  const int value = static_cast<int>(stereo);
  if (value < 0 || value > 5) {
    throw std::runtime_error("bond stereo is not in the PSICHIC e_map");
  }
  return value;
}

ROMol mol_from_pickle(const std::string &raw) {
  ROMol mol;
  MolPickler::molFromPickle(raw, mol);
  if (mol.needsUpdatePropertyCache()) {
    mol.updatePropertyCache(false);
  }
  return mol;
}

void ensure_rings(ROMol &mol) {
  if (!mol.getRingInfo()->isInitialized()) {
    MolOps::findSSSR(mol);
  }
}

struct FromRdmolArrays {
  int n_atoms = 0;
  std::vector<int64_t> atom_ids;
  std::vector<int64_t> atomic_nums;
  std::vector<int64_t> xs;
  std::vector<int64_t> edge_index;
  std::vector<int64_t> edge_attr;
  size_t n_edges = 0;
};

struct TreePartsArrays {
  int n_cliques = 0;
  size_t n_tree_edges = 0;
  std::vector<int64_t> xs;
  std::vector<int64_t> atom2clique_row;
  std::vector<int64_t> atom2clique_col;
  std::vector<int64_t> edge_c1;
  std::vector<int64_t> edge_c2;
  std::vector<int64_t> edge_w;
  std::vector<int64_t> tree_index;
};

void fill_kruskal_tree(TreePartsArrays &out);

FromRdmolArrays compute_from_rdmol(const ROMol &mol) {
  FromRdmolArrays out;
  const int n_atoms = static_cast<int>(mol.getNumAtoms());
  const int n_bonds = static_cast<int>(mol.getNumBonds());
  out.n_atoms = n_atoms;
  out.atom_ids.resize(n_atoms);
  out.atomic_nums.resize(n_atoms);
  out.xs.resize(static_cast<size_t>(n_atoms) * 8);
  for (int i = 0; i < n_atoms; ++i) {
    const Atom *atom = mol.getAtomWithIdx(i);
    const int atomic_num = static_cast<int>(atom->getAtomicNum());
    out.atom_ids[i] = map_atom_code(atomic_num);
    out.atomic_nums[i] = atomic_num;
    int64_t *row = out.xs.data() + static_cast<size_t>(i) * 8;
    row[0] = chirality_index(atom->getChiralTag());
    row[1] = degree_index(atom->getTotalDegree());
    row[2] = charge_index(atom->getFormalCharge());
    row[3] = hs_index(atom->getTotalNumHs());
    row[4] = radical_index(atom->getNumRadicalElectrons());
    row[5] = hybridization_index(atom->getHybridization());
    row[6] = atom->getIsAromatic() ? 1 : 0;
    row[7] = mol.getRingInfo()->numAtomRings(i) > 0 ? 1 : 0;
  }

  struct Directed {
    int64_t i;
    int64_t j;
    int64_t a0;
    int64_t a1;
    int64_t a2;
    int64_t key;
  };
  std::vector<Directed> edges;
  edges.reserve(static_cast<size_t>(n_bonds) * 2);
  for (int b = 0; b < n_bonds; ++b) {
    const Bond *bond = mol.getBondWithIdx(b);
    const int64_t i = bond->getBeginAtomIdx();
    const int64_t j = bond->getEndAtomIdx();
    const int64_t a0 = bond_type_index(bond->getBondType());
    const int64_t a1 = stereo_index(bond->getStereo());
    const int64_t a2 = bond->getIsConjugated() ? 1 : 0;
    edges.push_back({i, j, a0, a1, a2, i * n_atoms + j});
    edges.push_back({j, i, a0, a1, a2, j * n_atoms + i});
  }
  std::stable_sort(edges.begin(), edges.end(),
                   [](const Directed &a, const Directed &b) {
                     return a.key < b.key;
                   });
  out.n_edges = edges.size();
  out.edge_index.resize(out.n_edges * 2);
  out.edge_attr.resize(out.n_edges * 3);
  for (size_t e = 0; e < out.n_edges; ++e) {
    out.edge_index[e] = edges[e].i;
    out.edge_index[out.n_edges + e] = edges[e].j;
    out.edge_attr[e * 3] = edges[e].a0;
    out.edge_attr[e * 3 + 1] = edges[e].a1;
    out.edge_attr[e * 3 + 2] = edges[e].a2;
  }
  return out;
}

int intersect_count(const std::vector<int> &a, const std::vector<int> &b,
                    std::vector<int> &stamp, int &gen) {
  if (++gen == 0) {
    std::fill(stamp.begin(), stamp.end(), 0);
    gen = 1;
  }
  for (int atom : a) {
    stamp[atom] = gen;
  }
  int count = 0;
  for (int atom : b) {
    if (stamp[atom] == gen) {
      ++count;
    }
  }
  return count;
}

std::vector<int> ordered_union(const std::vector<int> &left,
                               const std::vector<int> &right,
                               std::vector<int> &stamp, int &gen) {
  if (++gen == 0) {
    std::fill(stamp.begin(), stamp.end(), 0);
    gen = 1;
  }
  std::vector<int> out;
  out.reserve(left.size() + right.size());
  for (int atom : left) {
    if (stamp[atom] != gen) {
      stamp[atom] = gen;
      out.push_back(atom);
    }
  }
  for (int atom : right) {
    if (stamp[atom] != gen) {
      stamp[atom] = gen;
      out.push_back(atom);
    }
  }
  return out;
}

TreePartsArrays compute_tree_parts(ROMol &mol) {
  TreePartsArrays out;
  MolOps::symmetrizeSSSR(mol);
  const int n_atoms = static_cast<int>(mol.getNumAtoms());
  const auto &rings = mol.getRingInfo()->atomRings();
  std::vector<std::vector<int>> cliques;
  std::vector<int> xs;
  cliques.reserve(rings.size() + mol.getNumBonds() + n_atoms);
  xs.reserve(rings.size() + mol.getNumBonds() + n_atoms);
  for (const auto &ring : rings) {
    cliques.emplace_back(ring.begin(), ring.end());
    xs.push_back(0);
  }
  for (const Bond *bond : mol.bonds()) {
    if (mol.getRingInfo()->numBondRings(bond->getIdx()) == 0) {
      cliques.push_back({static_cast<int>(bond->getBeginAtomIdx()),
                         static_cast<int>(bond->getEndAtomIdx())});
      xs.push_back(1);
    }
  }

  std::vector<std::vector<int>> atom2clique(n_atoms);
  for (int c = 0; c < static_cast<int>(cliques.size()); ++c) {
    for (int atom : cliques[c]) {
      atom2clique[atom].push_back(c);
    }
  }

  std::vector<int> stamp(n_atoms, 0);
  int gen = 0;
  const int n_initial = static_cast<int>(cliques.size());
  for (int c1 = 0; c1 < n_initial; ++c1) {
    const std::vector<int> atoms = cliques[c1];
    for (int atom : atoms) {
      for (int c2 : atom2clique[atom]) {
        if (c1 >= c2 || static_cast<int>(cliques[c1].size()) <= 2 ||
            static_cast<int>(cliques[c2].size()) <= 2) {
          continue;
        }
        if (intersect_count(cliques[c1], cliques[c2], stamp, gen) > 2) {
          cliques[c1] = ordered_union(cliques[c1], cliques[c2], stamp, gen);
          xs[c1] = 2;
          cliques[c2].clear();
          xs[c2] = -1;
        }
      }
    }
  }

  std::vector<std::vector<int>> kept;
  std::vector<int> kept_xs;
  kept.reserve(cliques.size());
  kept_xs.reserve(xs.size());
  for (size_t i = 0; i < cliques.size(); ++i) {
    if (!cliques[i].empty()) {
      kept.push_back(std::move(cliques[i]));
      kept_xs.push_back(xs[i]);
    }
  }
  cliques.swap(kept);
  xs.swap(kept_xs);

  atom2clique.assign(n_atoms, {});
  for (int c = 0; c < static_cast<int>(cliques.size()); ++c) {
    for (int atom : cliques[c]) {
      atom2clique[atom].push_back(c);
    }
  }

  std::unordered_map<uint64_t, int> edge_index;
  auto set_edge = [&](int c1, int c2, int weight) {
    const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(c1)) << 32) |
                         static_cast<uint32_t>(c2);
    auto found = edge_index.find(key);
    if (found == edge_index.end()) {
      edge_index[key] = static_cast<int>(out.edge_c1.size());
      out.edge_c1.push_back(c1);
      out.edge_c2.push_back(c2);
      out.edge_w.push_back(weight);
    } else {
      out.edge_w[found->second] = weight;
    }
  };

  for (int atom = 0; atom < n_atoms; ++atom) {
    const std::vector<int> &cs = atom2clique[atom];
    if (cs.size() <= 1) {
      continue;
    }
    std::vector<int> bonds;
    std::vector<int> rings_here;
    for (int c : cs) {
      if (cliques[c].size() == 2) {
        bonds.push_back(c);
      }
      if (cliques[c].size() > 4) {
        rings_here.push_back(c);
      }
    }
    if (bonds.size() > 2 || (bonds.size() == 2 && cs.size() > 2)) {
      cliques.push_back({atom});
      xs.push_back(3);
      const int c2 = static_cast<int>(cliques.size()) - 1;
      for (int c1 : cs) {
        set_edge(c1, c2, 1);
      }
    } else if (rings_here.size() > 2) {
      cliques.push_back({atom});
      xs.push_back(3);
      const int c2 = static_cast<int>(cliques.size()) - 1;
      for (int c1 : cs) {
        set_edge(c1, c2, 99);
      }
    } else {
      for (size_t i = 0; i < cs.size(); ++i) {
        for (size_t j = i + 1; j < cs.size(); ++j) {
          const int c1 = cs[i];
          const int c2 = cs[j];
          const int count = intersect_count(cliques[c1], cliques[c2], stamp, gen);
          const uint64_t key =
              (static_cast<uint64_t>(static_cast<uint32_t>(c1)) << 32) |
              static_cast<uint32_t>(c2);
          auto found = edge_index.find(key);
          const int previous =
              found == edge_index.end() ? 99 : static_cast<int>(out.edge_w[found->second]);
          set_edge(c1, c2, std::min(count, previous));
        }
      }
    }
  }

  atom2clique.assign(n_atoms, {});
  for (int c = 0; c < static_cast<int>(cliques.size()); ++c) {
    for (int atom : cliques[c]) {
      atom2clique[atom].push_back(c);
    }
  }
  for (int atom = 0; atom < n_atoms; ++atom) {
    for (int c : atom2clique[atom]) {
      out.atom2clique_row.push_back(atom);
      out.atom2clique_col.push_back(c);
    }
  }
  out.n_cliques = static_cast<int>(cliques.size());
  out.xs.assign(xs.begin(), xs.end());
  fill_kruskal_tree(out);
  return out;
}

void fill_kruskal_tree(TreePartsArrays &out) {
  // Same (100-w, a, b) sort, parent[pa]=pb find, and unique bidirectional
  // codes as the Python Kruskal in H27. tree_index is (2, n_edges) C-order.
  const int n = out.n_cliques;
  out.tree_index.clear();
  out.n_tree_edges = 0;
  if (n <= 0 || out.edge_c1.empty()) {
    return;
  }
  struct Item {
    int key_w;
    int a;
    int b;
  };
  std::vector<Item> items;
  items.reserve(out.edge_c1.size());
  for (size_t i = 0; i < out.edge_c1.size(); ++i) {
    items.push_back({100 - static_cast<int>(out.edge_w[i]),
                     static_cast<int>(out.edge_c1[i]),
                     static_cast<int>(out.edge_c2[i])});
  }
  std::sort(items.begin(), items.end(), [](const Item &left, const Item &right) {
    if (left.key_w != right.key_w) {
      return left.key_w < right.key_w;
    }
    if (left.a != right.a) {
      return left.a < right.a;
    }
    return left.b < right.b;
  });
  std::vector<int> parent(static_cast<size_t>(n));
  std::iota(parent.begin(), parent.end(), 0);
  auto find = [&](int idx) {
    while (parent[idx] != idx) {
      parent[idx] = parent[parent[idx]];
      idx = parent[idx];
    }
    return idx;
  };
  std::vector<int64_t> codes;
  codes.reserve(items.size() * 2);
  const int64_t nn = n;
  for (const Item &item : items) {
    const int pa = find(item.a);
    const int pb = find(item.b);
    if (pa == pb) {
      continue;
    }
    parent[pa] = pb;
    codes.push_back(static_cast<int64_t>(item.a) * nn + item.b);
    codes.push_back(static_cast<int64_t>(item.b) * nn + item.a);
  }
  std::sort(codes.begin(), codes.end());
  codes.erase(std::unique(codes.begin(), codes.end()), codes.end());
  out.n_tree_edges = codes.size();
  out.tree_index.resize(out.n_tree_edges * 2);
  for (size_t e = 0; e < out.n_tree_edges; ++e) {
    out.tree_index[e] = codes[e] / nn;
    out.tree_index[out.n_tree_edges + e] = codes[e] % nn;
  }
}

void fill_from_rdmol_dict(py::dict &out, const FromRdmolArrays &arrays) {
  out["atom_ids"] = vec_array(arrays.atom_ids);
  out["atomic_nums"] = vec_array(arrays.atomic_nums);
  out["x"] = mat_array(arrays.n_atoms, 8, arrays.xs);
  out["edge_index"] = mat_array(2, arrays.n_edges, arrays.edge_index);
  out["edge_attr"] = mat_array(arrays.n_edges, 3, arrays.edge_attr);
}

void fill_tree_dict(py::dict &out, const TreePartsArrays &tree) {
  out["n_cliques"] = tree.n_cliques;
  out["xs"] = vec_array(tree.xs);
  out["atom2clique_row"] = vec_array(tree.atom2clique_row);
  out["atom2clique_col"] = vec_array(tree.atom2clique_col);
  out["edge_c1"] = vec_array(tree.edge_c1);
  out["edge_c2"] = vec_array(tree.edge_c2);
  out["edge_w"] = vec_array(tree.edge_w);
  out["tree_index"] = mat_array(2, tree.n_tree_edges, tree.tree_index);
}

py::dict from_rdmol(const py::bytes &pickle) {
  std::string raw = pickle;
  FromRdmolArrays arrays;
  {
    py::gil_scoped_release release;
    ROMol mol = mol_from_pickle(raw);
    ensure_rings(mol);
    arrays = compute_from_rdmol(mol);
  }
  py::dict out;
  fill_from_rdmol_dict(out, arrays);
  return out;
}

py::dict tree_parts(const py::bytes &pickle) {
  std::string raw = pickle;
  TreePartsArrays tree;
  {
    py::gil_scoped_release release;
    ROMol mol = mol_from_pickle(raw);
    tree = compute_tree_parts(mol);
  }
  py::dict out;
  fill_tree_dict(out, tree);
  return out;
}

py::dict molecule_parts(const py::bytes &pickle) {
  std::string raw = pickle;
  FromRdmolArrays arrays;
  TreePartsArrays tree;
  {
    py::gil_scoped_release release;
    ROMol mol = mol_from_pickle(raw);
    ensure_rings(mol);
    arrays = compute_from_rdmol(mol);
    tree = compute_tree_parts(mol);
  }
  py::dict out;
  fill_from_rdmol_dict(out, arrays);
  fill_tree_dict(out, tree);
  return out;
}

py::object molecule_from_smiles(const std::string &smi) {
  FromRdmolArrays arrays;
  TreePartsArrays tree;
  bool ok = false;
  {
    py::gil_scoped_release release;
    std::unique_ptr<RWMol> mol(SmilesToMol(smi));
    if (mol) {
      ensure_rings(*mol);
      arrays = compute_from_rdmol(*mol);
      tree = compute_tree_parts(*mol);
      ok = true;
    }
  }
  if (!ok) {
    return py::none();
  }
  py::dict out;
  fill_from_rdmol_dict(out, arrays);
  fill_tree_dict(out, tree);
  return out;
}

}  // namespace

PYBIND11_MODULE(graph_ext, m) {
  m.def("from_rdmol", &from_rdmol, py::arg("pickle"));
  m.def("tree_parts", &tree_parts, py::arg("pickle"));
  m.def("molecule_parts", &molecule_parts, py::arg("pickle"));
  m.def("molecule_from_smiles", &molecule_from_smiles, py::arg("smiles"));
}
