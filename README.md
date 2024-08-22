# CATPCA

## Autor: Lazar Kračunović

### Pregled Projekta

Cilj ovog projekta je bila implementacija dve varijante PCA algoritma koje proširuju standardni PCA na ordinalne i nominalne podatke, koji se ne mogu adekvatno obraditi korišćenjem standardnog PCA algoritma. Konkretno, reč je o algoritmima PRINCALS i PRINCIPALS, koji se zasnivaju na ALS algoritmu (Alternating Least Squares). Ovi algoritmi funkcionišu na principu alternacije između kvantizacije kategorijskih promenljivih i izračunavanja scores objekata.

U Python skriptama koje nose nazive ovih algoritama implementirani su ovi algoritmi, bazirani na radu WIREs Comput Stat 2013, 5:456–464. doi: 10.1002/wics.1279. Takođe, u Python sveskama su ovi algoritmi primenjeni na Guttman-Bellovim podacima.
