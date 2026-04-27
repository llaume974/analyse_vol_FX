🔹 FX Volatility Surface Analysis (Desk-Oriented Version)

Objectif du projet :
Construire, analyser et exploiter une surface de volatilité implicite FX afin d’en extraire des signaux de marché, opportunités de trading et insights pour pricing de produits structurés.

🔹 1. Construction de la surface (comme sur un desk)
Récupération de vraies données de marché (ATM vol, Risk Reversal, Butterfly) sur plusieurs maturités (1W → 1Y+) (on peut ici recuperer des données sur EUR/USD)
Conversion des quotes FX standards en volatilités par strike :
ATM → niveau central
RR → skew (déséquilibre call vs put)
BF → convexité (smile)

👉 Reconstruction complète de la surface implicite :

Interpolation en maturité (spline / linear / SABR-like)
Interpolation en strike (smile fitting)
🔹 2. Analyse du smile et du skew (logique trader)
➤ Étude du skew (Risk Reversal)
Analyse du 25Δ RR pour capter :
biais directionnel du marché (ex : demande de protection downside USD)
positioning des investisseurs

👉 Exemple d’insight desk :

RR EUR/USD > 0 → marché paie les calls EUR → biais haussier EUR
RR très négatif → stress USD → flight to quality
➤ Étude du smile (Butterfly)
Analyse du BF pour mesurer :
demande de convexité
pricing des extrêmes (tail risk)

👉 Insight :

BF élevé → marché anticipe des mouvements extrêmes
BF bas → marché “complacent”
🔹 3. Dynamique temporelle (ce que fait vraiment un desk)
Tracking de la surface dans le temps
Analyse des variations :
shift parallèle → mouvement global de vol
twist → changement de skew
smile deformation → pricing du risk tail

👉 Tu passes de :
“j’ai une surface”
à
“je comprends comment le marché se repositionne”

🔹 4. Applications directes desk (très important)
➤ Pricing produits structurés
Impact du skew sur :
collars
risk reversals
barriers

👉 Exemple :

Skew élevé → vendre des puts est plus rémunérateur → structuration optimisée
➤ Trading / Relative Value
Détection de dislocations :
skew trop élevé vs historique
BF anormalement bas/haut

👉 Stratégies :

Long/short skew (RR trades)
Long convexity (acheter BF)
Arbitrage surface (calendar / strike)
➤ Hedging
Compréhension de :
vega exposure
skew risk (non-linéarité)
🔹 5. Output concret (ce que tu montres en entretien)
Heatmap de la surface (strike vs maturité)
Graphs :
smile par maturité
skew (RR) dans le temps
Dashboard :
niveaux actuels vs historique
z-score de skew / BF
🔹 6. Conclusion type desk (ULTRA important)

👉 Exemple de conclusion que tu peux dire :

“This project allowed me to move from a purely theoretical understanding of volatility to a market-oriented approach. By reconstructing the FX volatility surface and analyzing skew and smile dynamics, I was able to identify how the market prices directional risk and tail events, and how these elements directly impact structuring decisions and relative value opportunities on a trading desk.”