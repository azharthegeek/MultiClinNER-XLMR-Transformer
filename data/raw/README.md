# Dataset: MultiClinAI

## Download

Request access to the MultiClinAI Shared Task dataset via:
- Zenodo release (search "MultiClinAI NER")
- Shared task website / organizers

## Required File Placement

Once downloaded, place the CoNLL files as follows:

```
data/raw/
├── en/
│   ├── train.conll
│   ├── dev.conll
│   └── test.conll
└── es/
    ├── train.conll
    ├── dev.conll
    └── test.conll
```

## File Format

One token per line, tab-separated: `token\tBIO_tag`  
Blank line = sentence boundary (CoNLL-2003 standard).

```
Patient    O
presented  O
with       O
chest      B-SYM
pain       I-SYM
.          O

El         O
paciente   O
presenta   O
fiebre     B-SYM
.          O
```

## Entity Labels

| Label | Meaning |
|-------|---------|
| O | Outside any entity |
| B-DIS | Beginning of Disease |
| I-DIS | Inside of Disease |
| B-SYM | Beginning of Symptom |
| I-SYM | Inside of Symptom |
| B-PRO | Beginning of Procedure |
| I-PRO | Inside of Procedure |
