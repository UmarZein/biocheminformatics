from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

import math
import pickle

import os.path as op

_fscores = None
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)


def readFragmentScores(name="fpscores.pkl.gz"):
  import gzip
  global _fscores
  # generate the full path filename:
  if name == "fpscores.pkl.gz":
    name = op.join(op.dirname(__file__), name)
  data = pickle.load(gzip.open(name))
  outDict = {}
  for i in data:
    for j in range(1, len(i)):
      outDict[i[j]] = float(i[0])
  _fscores = outDict


def numBridgeheadsAndSpiro(mol, ri=None):
  nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
  nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
  return nBridgehead, nSpiro


def calculateScore(m):

  if not m.GetNumAtoms():
    return None

  if _fscores is None:
    readFragmentScores()

  # fragment score
  sfp = mfpgen.GetSparseCountFingerprint(m)

  score1 = 0.
  nf = 0
  nze = sfp.GetNonzeroElements()
  for id, count in nze.items():
    nf += count
    score1 += _fscores.get(id, -4) * count

  score1 /= nf

  # features score
  nAtoms = m.GetNumAtoms()
  nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
  ri = m.GetRingInfo()
  nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
  nMacrocycles = 0
  for x in ri.AtomRings():
    if len(x) > 8:
      nMacrocycles += 1

  sizePenalty = nAtoms**1.005 - nAtoms
  stereoPenalty = math.log10(nChiralCenters + 1)
  spiroPenalty = math.log10(nSpiro + 1)
  bridgePenalty = math.log10(nBridgeheads + 1)
  macrocyclePenalty = 0.
  # ---------------------------------------
  # This differs from the paper, which defines:
  #  macrocyclePenalty = math.log10(nMacrocycles+1)
  # This form generates better results when 2 or more macrocycles are present
  if nMacrocycles > 0:
    macrocyclePenalty = math.log10(2)

  score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

  # correction for the fingerprint density
  # not in the original publication, added in version 1.1
  # to make highly symmetrical molecules easier to synthetise
  score3 = 0.
  numBits = len(nze)
  if nAtoms > numBits:
    score3 = math.log(float(nAtoms) / numBits) * .5

  sascore = score1 + score2 + score3

  # need to transform "raw" value into scale between 1 and 10
  min = -4.0
  max = 2.5
  sascore = 11. - (sascore - min + 1) / (max - min) * 9.

  # smooth the 10-end
  if sascore > 8.:
    sascore = 8. + math.log(sascore + 1. - 9.)
  if sascore > 10.:
    sascore = 10.0
  elif sascore < 1.:
    sascore = 1.0

  return sascore