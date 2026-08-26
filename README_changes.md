# Changes made to original codebae

### `hgraph/chemutils.py` <br/>
Issue:
The crash occurs because RDKit’s newer version fails to compute resonance structures cleanly for intermediate fragments
<br/>
<br/>
old
```
def get_mol(smiles):
     mol = Chem.MolFromSmiles(smiles)
     if mol is not None: Chem.Kekulize(mol)
     return mol
```
new
```
def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        try:
            # Re-enabling with clearAromaticFlags cleans up the 
            # sub-fragment's valence states natively so kekulization works.
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            pass # Fallback safety for completely unresolvable sub-fragments
    return mol
```

### `prerocess.py` <br/>
Issue:
Number of splits was zero: hence division by zero
<br/>
<br/>