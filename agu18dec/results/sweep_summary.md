# Phase-2 sweep summary

Phase-1 baseline Student (aux. only) = **0.537**

## m_ghost

| value | student_ghost | ±CI | xmodel_ghost | teacher |
|---|---|---|---|---|
| 1 | 0.103 | 0.004 | 0.103 | 0.943 |
| 2 | 0.348 | 0.018 | 0.106 | 0.943 |
| 3 | 0.537 | 0.022 | 0.100 | 0.943 |
| 5 | 0.704 | 0.016 | 0.109 | 0.943 |
| 10 | 0.835 | 0.011 | 0.098 | 0.943 |
| 20 | 0.894 | 0.005 | 0.105 | 0.943 |
| 50 | 0.917 | 0.002 | 0.108 | 0.943 |

## epochs

| value | student_ghost | ±CI | xmodel_ghost | teacher |
|---|---|---|---|---|
| 5 | 0.537 | 0.022 | 0.100 | 0.943 |

## width

| value | student_ghost | ±CI | xmodel_ghost | teacher |
|---|---|---|---|---|
| (128, 128) | 0.528 | 0.021 | 0.110 | 0.929 |
| (256, 256) | 0.537 | 0.022 | 0.100 | 0.943 |

---
**Best single config:** m_ghost=50 -> aux-only 0.917