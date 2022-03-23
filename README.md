# InfoShield: Generalizable Information-Theoretic Human-Trafficking Detection

------------

Lee, M.C., Vajiac, C., Kulshrestha, A., Levy, S., Park, N., Jones, C., Rabbany, R., and Faloutsos, C., "InfoShield: Generalizable Information-Theoretic Human-Trafficking Detection". *37th IEEE International Conference on Data Engineering (ICDE)*, 2021.

Please cite the paper as:

    @inproceedings{lee2021InfoShield,
      title={{InfoShield:} Generalizable Information-Theoretic Human-Trafficking Detection},
      author={Lee, Meng-Chieh and Vajiac, Catalina and Kulshrestha, Aayushi and Levy, Sacha and Park, Namyong and Jones, Cara and Rabbany, Reihaneh and Faloutsos, Christos},
      booktitle={2021 37th IEEE International Conference on Data Engineering (ICDE)},
      year={2021},
      organization={IEEE},
    }
    
##  Introduction
In this paper, we present INFOSHIELD, which makes the following contributions:
- **Practical**: being scalable and effective on real data
- **Parameter-free and Principled**: requiring no user-defined parameters
- **Interpretable**: finding a document to be the cluster representative, highlighting all the common phrases, and automatically detecting “slots”, i.e. phrases that differ in every document
- **Generalizable**: beating or matching domainspecific methods in Twitter bot detection and human trafficking detection respectively, as well as being language-independent finding clusters in Spanish, Italian, and Japanese.

## Usage

To run the InfoShield demo:
`make demo`

To specify the column headers for unique id (id_str) and text (text_str):
`python infoshield.py CSV_FILENAME id_str text_str`

To run InfoShield:
`python infoshield.py data/sample_input.csv id text`

To run InfoShield-Coarse only:
`python infoshieldcoarse.py CSV_FILENAME`

To run InfoShield-Fine only:
`python infoshieldfine.py CSV_REUSLT_FROM_COARSE`

## Acknowledgement
One part of our code is based on Partial Order Alignment, downloaded from https://github.com/ljdursi/poapy.

This implementation is according to the following paper:

Lee, C., Grasso, C., & Sharlow, M. F. (2002). Multiple sequence alignment using partial order graphs. Bioinformatics, 18(3), 452-464.
