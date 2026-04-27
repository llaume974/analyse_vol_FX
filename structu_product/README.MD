## Projet 4 — FX Structured Products Pricer lié à la surface de volatilité

L’idée : créer un outil qui permet à un desk FX de **pricer et comparer des produits structurés simples** à partir d’une **surface de volatilité FX**.

Le but n’est pas juste de calculer un payoff. Le vrai intérêt est de répondre à une question desk :

> “Quel produit proposer au client, à quel strike, avec quel coût, et comment la surface de volatilité influence le prix ?”

---

## Objectif du projet

Tu construis un outil capable de pricer :

* **vanilla call / put FX**
* **forward**
* **collar**
* **risk reversal**
* **seagull**
* éventuellement **participating forward**
* éventuellement **target redemption forward simplifié**

L’outil prend en entrée :

* spot EUR/USD
* taux EUR
* taux USD
* maturité
* notionnel
* strikes
* volatilités implicites issues de la surface
* sens client : importateur/exportateur
* devise à couvrir

Et il sort :

* prix / prime
* payoff chart
* break-even
* coût du hedge
* protection minimale
* upside abandonné
* sensibilité au spot
* sensibilité à la volatilité
* impact du skew
* comparaison entre structures

---

## Cas pratique desk

Exemple :

> Une entreprise européenne doit payer 10M USD dans 3 mois. Elle craint une baisse de EUR/USD, donc un USD plus fort.

L’outil doit pouvoir dire :

* Forward : hedge complet mais aucun upside.
* Call USD / put EUR : protection flexible mais prime à payer.
* Collar : protection financée par abandon d’une partie du upside.
* Seagull : améliore le coût mais ajoute un risque de queue.
* Risk reversal : très sensible au skew de volatilité.

---

## Architecture du projet

### 1. Page “Market Inputs”

Tu mets les données de marché :

| Input    |        Exemple |
| -------- | -------------: |
| Pair     |        EUR/USD |
| Spot     |         1.1500 |
| Maturity |             3M |
| EUR rate |          2.40% |
| USD rate |          4.60% |
| Notional | 10,000,000 USD |
| ATM vol  |          7.50% |
| 25D RR   |  -0.20 vol pts |
| 25D BF   |   0.28 vol pts |

Tu reconstruis ensuite :

```text
25D Call vol = ATM + BF + RR/2
25D Put vol  = ATM + BF - RR/2
```

Exemple si ATM = 7.50%, RR = -0.20%, BF = 0.28% :

```text
25D Call vol = 7.50% + 0.28% - 0.10% = 7.68%
25D Put vol  = 7.50% + 0.28% + 0.10% = 7.88%
```

Donc les puts EUR sont plus chers que les calls EUR : le marché paie davantage la protection contre une baisse de EUR/USD.

---

## 2. Pricing engine

Tu codes un moteur Black-Scholes FX, aussi appelé Garman-Kohlhagen.

Formule utilisée :

```text
Call = S * exp(-rf*T) * N(d1) - K * exp(-rd*T) * N(d2)
Put = K * exp(-rd*T) * N(-d2) - S * exp(-rf*T) * N(-d1)
```

Avec :

```text
d1 = [ln(S/K) + (rd - rf + 0.5σ²)T] / [σ√T]
d2 = d1 - σ√T
```

Pour EUR/USD :

* `S` = spot EUR/USD
* `rd` = taux USD
* `rf` = taux EUR
* le payoff est exprimé en USD par EUR

C’est important : pour EUR/USD, le domestic rate est souvent le taux USD, car le prix est coté en USD pour 1 EUR.

---

## 3. Produits à intégrer

### A. Forward FX

Le forward donne un taux garanti.

```text
F = S * exp((rd - rf)T)
```

Utilité desk :

> Permet de comparer tous les produits structurés à une couverture linéaire simple.

Décision client :

* hedge parfait
* zéro prime
* pas de participation favorable au marché

---

### B. Option vanilla

Pour une entreprise qui doit acheter des USD :

Elle veut se protéger contre une baisse de EUR/USD.

Donc elle peut acheter :

* un put EUR/USD
* ou équivalent : un call USD/EUR

Dans ton outil, il faut afficher :

* prime
* payoff
* strike de protection
* break-even
* coût en % du notionnel

Phrase desk :

> La vanilla offre une protection asymétrique mais son coût dépend fortement du niveau d’ATM vol et du skew.

---

### C. Collar

Un collar combine :

* achat d’un put de protection
* vente d’un call pour financer la prime

Pour une entreprise européenne qui doit acheter des USD :

* achat put EUR/USD à strike bas
* vente call EUR/USD à strike haut

Effet :

* protégée si EUR/USD baisse
* profite partiellement si EUR/USD monte
* mais upside capé au-dessus du call vendu

Objectif de l’outil :

trouver automatiquement le call strike qui rend la structure **zero-cost**.

```text
Premium put acheté ≈ Premium call vendu
```

Sortie utile :

> Pour une protection à 1.12, le call vendu doit être à 1.18 pour obtenir un collar zéro-cost.

---

### D. Risk Reversal

Un risk reversal est proche du collar, mais souvent construit avec des options 25 delta :

* achat 25D put
* vente 25D call

Il sert à exprimer ou couvrir le skew.

Dans ton outil :

* tu récupères les vols 25D put/call
* tu prices chaque jambe
* tu mesures l’impact du RR sur le coût

Message desk :

> Si le 25D put est plus cher que le 25D call, le hedge downside coûte plus cher à financer, donc le collar devient moins favorable pour l’acheteur de protection.

---

### E. Seagull

Le seagull ajoute une troisième option.

Exemple pour réduire le coût :

* achat put proche du spot
* vente call OTM
* vente put très OTM

Effet :

* protection sur une zone
* coût réduit
* mais perte de protection extrême au-delà d’un niveau très bas

Utilité desk :

> Produit intéressant pour un client qui veut réduire la prime, mais qui accepte un risque extrême au-delà d’un certain niveau.

Dans l’outil, tu affiches :

* zone protégée
* zone non protégée
* prime totale
* scénario défavorable
* payoff en cas de choc extrême

---

## 4. Surface de volatilité : vraie valeur ajoutée

Le gros point différenciant du projet, c’est que tu ne prices pas toutes les options avec la même vol.

Tu utilises :

* ATM vol pour les strikes proches du forward
* 25D put vol pour les puts OTM
* 25D call vol pour les calls OTM
* interpolation pour les strikes intermédiaires

Donc ton outil montre que :

> Deux options avec la même maturité mais des strikes différents n’ont pas le même niveau de volatilité implicite.

C’est ça qui rend le projet “desk”.

---

## 5. Analyse de sensibilité

Tu ajoutes des chocs :

### Spot shock

Exemple :

* EUR/USD -2%
* EUR/USD -5%
* EUR/USD +2%
* EUR/USD +5%

Tu observes :

* PnL option
* payoff client
* hedge effectiveness

### Vol shock

Exemple :

* ATM vol +1 vol point
* ATM vol -1 vol point

Tu observes :

* impact sur prime
* impact sur seagull/collar
* variation du coût client

### Skew shock

Exemple :

* RR devient plus négatif
* RR revient vers zéro

Tu observes :

* put de protection plus cher
* collar moins facilement zero-cost
* call vendu finance moins bien la protection

Phrase utile :

> Un skew plus négatif rend la protection contre la baisse d’EUR/USD plus chère, ce qui détériore les conditions de hedge pour un client importateur USD.

---

## 6. Outputs finaux du dashboard

Ton dashboard doit sortir un tableau comme ça :

| Structure     |  Prime | Protection | Upside | Risque extrême | Commentaire desk                |
| ------------- | -----: | ---------- | ------ | -------------- | ------------------------------- |
| Forward       |      0 | Totale     | Aucun  | Faible         | Simple mais bloque le taux      |
| Put vanilla   | Élevée | Forte      | Total  | Faible         | Flexible mais coûteux           |
| Collar        |     ~0 | Forte      | Capé   | Faible         | Bon compromis                   |
| Risk reversal |     ~0 | Forte      | Capé   | Dépend du skew | Très sensible au RR             |
| Seagull       | Faible | Partielle  | Capé   | Élevé          | Moins cher mais risque de queue |

---

## Lancer la V1 Streamlit

L'application se trouve dans `structu_product/app.py` et charge par défaut le fichier `../data.csv`.

```bash
cd structu_product
python3 -m pip install -r requirements.txt
streamlit run app.py
```

La V1 contient :

* interface en français avec deux pages : `Synthèse` et `Atelier` ;
* import CSV avec validation du format ;
* surface de volatilité desk `tenor x delta bucket` en 3D interactive ;
* heatmap 2D ;
* paramètres marché EUR/USD regroupés dans la barre latérale ;
* pricer Garman-Kohlhagen ;
* comparaison forward, vanilla, collar, risk reversal et seagull ;
* solveur de strike pour collar zero-cost ;
* payoff charts ;
* sensibilités spot, volatilité ATM et skew/RR ;
* recommandation desk simple.
