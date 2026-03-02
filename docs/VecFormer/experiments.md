# Dataset

## Datasets

# Performance

如果出现明显于预期过高或过低，则用 ↑↓ 表示

## 5\% Train Rate

|Dataset|Amazon|Reddit|photo|elliptic|t_finance|tolokers|questions
|-|-|-|-|-|-|-|-|
|DOMINANT|0.8804|0.5667↑|0.5227|0.3230|--|0.5883↑|0.5906↑
|AnomalyDAE|0.9088|0.5209|0.6725↑|--|0.8146↑|0.5920|0.5860↑
|OCGNN|0.9076↑|0.5324|0.6473|0.2618|0.5501|0.4864|0.5382
|AEGIS|0.7516↑|0.5666|0.5954|0.6680|0.6561|0.4436|0.5101|
|GAAN|0.6530|0.5081↓|0.4290|-|-|0.3692↑|-
|TAM|0.8793↑|0.5910|0.5780|-|-|0.4805|-
|GGAD|0.6540|0.6143|0.6490|0.7006|0.7853|0.5382|0.4879|
|RHO|0.9439|0.6026|0.7587|0.8497||0.6436|0.5844
|VecFormer|0.9344|0.5782|0.8377|0.7475|0.8988|0.6509|0.5842

AP:

|Dataset|Amazon|Reddit|photo|elliptic|t_finance|tolokers|questions
|-|-|-|-|-|-|-|-|
|DOMINANT|0.7832↑|0.0398|0.1192|0.0670|⌛️|0.2831↑|0.0560↑
|AnomalyDAE|0.7171|0.0361|0.1450↑|--|0.1758↑|0.2708↑|0.0612↑
|OCGNN|0.7065|0.0340|0.1354|0.0623|0.0474|0.2142|0.0374
|AEGIS|0.2926|0.0398|0.1265↑|0.0663|0.0668|0.2007|0.0334|
|GAAN|0.0887|0.0345|0.0760|-|⌛️|0.1643|-
|TAM|0.6960↑|0.0431|0.1031|-|⌛️|0.2139|⌛️
|GGAD|0.1136|0.0502|0.1426|0.2565|0.1278|0.2449|0.0304
|RHO|0.8022|0.0577|0.2275|0.5088||0.3245|0.0432
|VecFormer|0.8033|0.0441|0.6074|0.2889|0.6448|0.3051|0.0396

GPU Support:

|Dataset|Amazon|Reddit|photo|elliptic|t_finance|tolokers|questions
|-|-|-|-|-|-|-|-|
|DOMINANT|✅|✅|✅|✅|✅|✅|✅
|AnomalyDAE|✅|✅|✅|❌|❌|✅|✅
|OCGNN|✅|✅|✅|✅|✅|✅|✅
|AEGIS|✅|✅|✅|-|✅|✅|✅|
|GAAN|✅|✅|✅|❌|❌|✅|❌
|TAM|✅|✅|✅|❌|❌|✅|❌
|GGAD|✅|✅|✅|❌|✅|✅|❌
|VecFormer|✅|✅|✅|✅|✅|✅|✅|✅|

### TBD

- DOMINANT:
    - t_finance: quiet-serenity-66295 (81 GPU 0, ETC 10-24 8:00)
- anomalyDAE
    - elliptic: legendary-lion-66542 (81 CPU, ETC 10-21 8:00)
- GAAN:
- TAM:
    - t_finance: 
- VecFormer: 
    - questions: pjd8tz5l

### To Reproduce:

#### DOMINANT

- Amazon: glorious-frost-57831
- Reddit: dashing-hill-58679
- photo: whole-plasma-57771
- elliptic: major-rain-66424
- t_finance: -
- tolokers: atomic-silence-65964
- questions: fine-water-65987

#### AnomalyDAE

- Amazon: cerulean-vortex-65991
- photo:  true-morning-65990
- Reddit: ancient-lion-65992
- elliptic: -
- t_finance: jolly-glade-66297
- tolokers: solar-sweep-5
- questions: glorious-sweep-2 (epoch 350)

#### OCGNN

- Amazon: fast-shape-66259
- reddit: graceful-leaf-66260
- photo: jumping-dew-66262
- elliptic: charmed-forest-66425 (epoch 700)
- t_finance: stellar-snowball-66422 (epoch 200)
- tolokers: jumping-dawn-66265
- questions: copper-energy-66427

#### AEGIS

- Amazon: wild-energy-66269
- reddit: dark-rain-66272
- photo: driven-totem-66271
- elliptic: -
- t_finance: likely-breeze-66421 (epoch 600)
- tolokers: soft-salad-66275
- questions: azure-haze-66423 (epoch 600)

#### GAAN

- Amazon: vocal-snow-66758 (epoch 700)
- reddit: rose-oath-66290
- photo: deep-wildflower-66289
- t_finance: young-vortex-66741 (epoch 300)
- tolokers: distinctive-smoke-66300

#### TAM

- Amazon: laced-sun-66432
- reddit: spring-wave-66416
- photo:generous-snow-66303
- tolokers: sparkling-shape-66435

#### GGAD

- reddit: clean-sound-66487
- photo: giddy-elevator-66470
- questions: different-sun-66462

#### VecFormer

- Amazon: 8ylmsq7q
- reddit: qs4t9byw
- photo: ck8br8o4
- elliptic: h6r7dlo2
- t_finance: iqxjqsdl
- tolokers: swl2tk10
- questions: 2nf7n8ds

# Parameter Study

## Performance w.r.t. Training Size

随着训练集比例变化，模型性能的变化。

AUC

|$R$|Amazon|Reddit|Photo|Elliptic|T-Finance|tolokers|
|-|-|-|-|-|-|-|
|$0.1\%$|0.8628|0.4681|0.7677|0.6023|**0.9038**|0.3754|
|$0.2\%$|0.7398|0.4902|0.8036|0.5862|0.8834|0.3345|
|$0.5\%$|**0.9456**|0.5584|0.8138|0.6243|0.8712|0.4414|
|$1\%$|0.9162|0.5349|**0.8403**|0.6520|0.8883|0.5856|
|$2\%$|0.9396|0.5630|0.8270|0.7284|0.8895|0.6309|
|$5\%$|0.9342|**0.5679**|0.8321|0.7502|0.8979|0.6463|
|$10\%$|0.9162|0.5670|0.8196|0.7495|0.8847|0.6713|
|$15\%$|0.9232|0.5546|0.8212|**0.7517**|0.8965|**0.6727**|

AP

|$R$|Amazon|Reddit|Photo|Elliptic|T-Finance|tolokers|
|-|-|-|-|-|-|-|
|$0.1\%$|0.1859|0.0295|0.3007|0.1164|0.5536|0.1644|
|$0.2\%$|0.5892|0.0326|0.2204|0.1087|0.5742|0.1539|
|$0.5\%$|**0.8302**|0.0442|0.5187|0.1238|0.4366|0.1863|
|$1\%$|0.7915|0.0400|0.5978|0.1586|0.5012|0.2614|
|$2\%$|0.8169|**0.0452**|0.5755|0.2359|0.5866|0.2999|
|$5\%$|0.8045|0.0426|**0.6069**|0.3026|**0.6376**|0.3029|
|$10\%$|0.8039|0.0443|0.5488|0.3076|0.5778|0.3206|
|$15\%$|0.7914|0.0440|0.5378|**0.3234**|0.6183|**0.3225**|

## Performance w.r.t. propagation steps

随着传播步数比例变化，模型性能的变化。


AUC

|$K$|Amazon|Reddit|Photo|Elliptic|T-Finance|tolokers|
|-|-|-|-|-|-|-|
|2|0.8816|0.4853|0.7790|0.6931|0.9166|0.5500|
|4|0.9114|0.4868|0.8357|0.7381|0.9024|0.5469|
|6|0.9094|0.4903|0.8337|0.7555|0.8993|0.5485|
|8|0.7318|0.4903|0.8496|0.7509|0.8822|0.5208|
|10|0.6886|0.5499|0.7826|0.7237|0.4121|0.4986|
|12|0.6181|0.5311|0.8500|0.7423|0.1658|0.5010|
|14|0.5697|0.5150|0.8129|0.7307|0.1685|0.5302|
|16|0.4436|0.5315|0.7986|0.6215|0.2153|0.5258|
|18|0.3566|0.5231|0.7011|0.6265|0.2422|0.5255|
|20|0.4002|0.5395|0.7105|0.5920|0.3556|0.5252|

AP

|$R$|Amazon|Reddit|Photo|Elliptic|T-Finance|tolokers|
|-|-|-|-|-|-|-|
|2|0.7809|0.0325|0.3867|0.1563|0.5942|0.2358|
|4|0.7943|0.0331|0.6022|0.2439|0.5830|0.2459|
|6|0.7510|0.0352|0.6065|0.3262|0.6179|0.2407|
|8|0.3187|0.0378|0.5385|0.3094|0.6156|0.2331|
|10|0.2781|0.0394|0.3666|0.2529|0.0771|0.2265|
|12|0.1321|0.0372|0.4212|0.2355|0.0256|0.2215|
|14|0.0889|0.0349|0.4252|0.2213|0.0258|0.2339|
|16|0.0582|0.0389|0.4202|0.1466|0.0273|0.2278|
|18|0.0511|0.0399|0.2982|0.1338|0.0280|0.2290|
|20|0.0505|0.0366|0.3741|0.1131|0.0350|0.2318|

## Performance w.r.t. propagation alpha

随着传播残差权重 alpha 变化，模型性能的变化。

AUC

|$\alpha$|Amazon|Reddit|Photo|Elliptic|T-Finance|tolokers|
|-|-|-|-|-|-|-|
|0|0.6661|0.5636|0.8323|0.5225|0.1726|0.5936|
|0.01|0.7215|0.5349|0.8520|0.5433|0.1744|0.5940|
|0.05|0.7914|0.5411|0.8960|0.5411|0.1755|0.5970|
|0.1|0.7088|0.5203|0.8924|0.4989|0.1875|0.6156|
|0.2|0.8901|0.5218|0.8383|0.4643|0.5505|0.6324|
|0.3|0.9156|0.5000|0.8200|0.5755|0.8981|0.6510|
|0.4|0.9159|0.4601|0.8098|0.7398|0.8997|0.6364|
|0.5|0.9150|0.4570|0.8144|0.7461|0.8976|0.6462|
|0.6|0.9051|0.4804|0.7557|0.7592|0.8872|0.6473|
|0.8|0.8907|0.4759|0.6586|0.7674|0.8862|0.6507|
|1|0.8891|0.4693|0.5574|0.7492|0.8779|0.6230|

AP

|$K$|Amazon|Reddit|Photo|Elliptic|T-Finance|tolokers|
|-|-|-|-|-|-|-|
|0|0.2535|0.0439|0.4610|0.0944|0.0259|0.2760|
|0.01|0.3525|0.0400|0.4931|0.0979|0.0260|0.2760|
|0.05|0.4670|0.0377|0.6210|0.0964|0.0259|0.2789|
|0.1|0.2589|0.0354|0.6332|0.0881|0.2622|0.2898|
|0.2|0.6931|0.0354|0.6086|0.0827|0.0856|0.2963|
|0.3|0.7899|0.0351|0.6051|0.1225|0.6377|0.3037|
|0.4|0.7900|0.0308|0.5578|0.2644|0.6186|0.2971|
|0.5|0.7960|0.0299|0.4480|0.3030|0.5964|0.2903|
|0.6|0.7938|0.0323|0.3271|0.2910|0.5343|0.2978|
|0.8|0.7904|0.0320|0.2034|0.2546|0.5128|0.3053|
|1|0.7886|0.0439|0.1162|0.2124|0.4138|0.3022|


## Computation Effiency

选取几个较大的数据集分析

||Tolokers|Amazon|questions|T-Finance|Elliptic
|-|-|-|-|-|-|
|Nodes|11,758|11,044|48,921|39,357|203,769
|Edges|519,000|4,398,392|153,540|21,222,543|234,355

||Tolokers|Amazon|questions|T-Finance|Elliptic
|-|-|-|-|-|-|
|GGAD|3830|3456|OOM|36410|OOM
|RHO|9120|6726|OOM||OOM
|VecFormer|6316|6340|23676|19170|22960
---

# 15\% By Myself

## 5\% Train Rate

|Dataset|Amazon|Reddit|photo|elliptic|t_finance|tolokers|questions
|-|-|-|-|-|-|-|-|
|AEGIS|0.6853|0.5949|0.6717|-|-|0.5568|-|
|GAAN|-|0.5334|0.4322|-|-|0.4917|-
|TAM|-|-|-|-|-|-|-


AP:

|Dataset|Amazon|Reddit|photo|elliptic|t_finance|tolokers|questions
|-|-|-|-|-|-|-|-|
|AEGIS|0.1648|0.0453|0.1632|-|-|0.2500|-|
|GAAN|-|0.0402|0.0778|-|-|0.2083|-

#### AEGIS

- Amazon: valiant-flower-66110 (epoch 600)
- reddit: sweet-sweep-5
- photo: decent-sweep-4
- elliptic: -
- t_finance: -
- tolokers: major-sweep-6
- questions: -

#### GAAN

- reddit: fast-wildflower-66195
- photo: tough-energy-66196 (epoch 100)
- tolokers: 