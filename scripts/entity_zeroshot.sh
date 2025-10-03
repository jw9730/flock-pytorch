# n = 16, ensemble = 16
# inductive entity
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d FB15k237Inductive:v1
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d FB15k237Inductive:v2
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d FB15k237Inductive:v3
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d FB15k237Inductive:v4
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d WN18RRInductive:v1
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d WN18RRInductive:v2
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d WN18RRInductive:v3
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d WN18RRInductive:v4
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d NELLInductive:v1
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d NELLInductive:v2
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d NELLInductive:v3
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d NELLInductive:v4
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d ILPC2022:small
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d HM:1k
# inductive entity-relation
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d FBIngram:25
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d FBIngram:50
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d FBIngram:75
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d FBIngram:100
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d WKIngram:25
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d WKIngram:50
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d WKIngram:75
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d WKIngram:100
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d NLIngram:0
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d NLIngram:25
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d NLIngram:50
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d NLIngram:75
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d NLIngram:100
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d WikiTopicsMT1:tax
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d WikiTopicsMT1:health
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d WikiTopicsMT2:sci
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d WikiTopicsMT4:sci
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d WikiTopicsMT4:health
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d Metafam
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d FBNELL
# transductive
python3 src_entity/run_many.py --config src_entity/config/zeroshot_transductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d WDsinger
python3 src_entity/run_many.py --config src_entity/config/zeroshot_transductive/n16_ensemble16.yaml --gpus [0] --ckpt $CKPT -d CoDExSmall

# n = 32, ensemble = 16
# inductive entity
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n32_ensemble16.yaml --gpus [0] --ckpt $CKPT -d HM:3k
# inductive entity-relation
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n32_ensemble16.yaml --gpus [0] --ckpt $CKPT -d WikiTopicsMT2:org
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n32_ensemble16.yaml --gpus [0] --ckpt $CKPT -d WikiTopicsMT3:art
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n32_ensemble16.yaml --gpus [0] --ckpt $CKPT -d WikiTopicsMT3:infra
# transductive
python3 src_entity/run_many.py --config src_entity/config/zeroshot_transductive/n32_ensemble16.yaml --gpus [0] --ckpt $CKPT -d NELL23k
python3 src_entity/run_many.py --config src_entity/config/zeroshot_transductive/n32_ensemble16.yaml --gpus [0] --ckpt $CKPT -d FB15k237_10

# n = 64, ensemble = 16
# inductive entity
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n64_ensemble16.yaml --gpus [0] --ckpt $CKPT -d ILPC2022:large
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n64_ensemble16.yaml --gpus [0] --ckpt $CKPT -d HM:5k
# transductive
python3 src_entity/run_many.py --config src_entity/config/zeroshot_transductive/n64_ensemble16.yaml --gpus [0] --ckpt $CKPT -d FB15k237_20
python3 src_entity/run_many.py --config src_entity/config/zeroshot_transductive/n64_ensemble16.yaml --gpus [0] --ckpt $CKPT -d FB15k237_50

# n = 128, ensemble = 16
# inductive entity
python3 src_entity/run_many.py --config src_entity/config/zeroshot_inductive/n128_ensemble16.yaml --gpus [0] --ckpt $CKPT -d HM:indigo
# transductive
python3 src_entity/run_many.py --config src_entity/config/zeroshot_transductive/n128_ensemble16.yaml --gpus [0] --ckpt $CKPT -d NELL995
python3 src_entity/run_many.py --config src_entity/config/zeroshot_transductive/n128_ensemble16.yaml --gpus [0] --ckpt $CKPT -d ConceptNet100k

# n = 256, ensemble = 4
# transductive
python3 src_entity/run_many.py --config src_entity/config/zeroshot_transductive/n256_ensemble4.yaml --gpus [0] --ckpt $CKPT -d AristoV4

# n = 512, ensemble = 8
# transductive
python3 src_entity/run_many.py --config src_entity/config/zeroshot_transductive/n512_ensemble8.yaml --gpus [0] --ckpt $CKPT -d YAGO310

# n = 512, ensemble = 2
# transductive
python3 src_entity/run_many.py --config src_entity/config/zeroshot_transductive/n512_ensemble2.yaml --gpus [0] --ckpt $CKPT -d CoDExLarge
python3 src_entity/run_many.py --config src_entity/config/zeroshot_transductive/n512_ensemble2.yaml --gpus [0] --ckpt $CKPT -d DBpedia100k
python3 src_entity/run_many.py --config src_entity/config/zeroshot_transductive/n512_ensemble2.yaml --gpus [0] --ckpt $CKPT -d Hetionet
