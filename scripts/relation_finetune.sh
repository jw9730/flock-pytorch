# n = 16, ensemble = 16
# inductive entity
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset FB15k237Inductive --version v1
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset FB15k237Inductive --version v2
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset FB15k237Inductive --version v3
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset FB15k237Inductive --version v4
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset WN18RRInductive --version v1
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset WN18RRInductive --version v2
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset WN18RRInductive --version v3
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset WN18RRInductive --version v4
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset NELLInductive --version v1
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset NELLInductive --version v2
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset NELLInductive --version v3
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset NELLInductive --version v4
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset ILPC2022 --version small
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset HM --version 1k
# inductive entity-relation
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset FBIngram --version 25
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset FBIngram --version 50
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset FBIngram --version 75
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset FBIngram --version 100
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset WKIngram --version 25
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset WKIngram --version 50
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset WKIngram --version 75
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset WKIngram --version 100
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset NLIngram --version 0
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset NLIngram --version 25
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset NLIngram --version 50
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset NLIngram --version 75
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset NLIngram --version 100
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset WikiTopicsMT1 --version tax
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset WikiTopicsMT1 --version health
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset WikiTopicsMT2 --version sci
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset WikiTopicsMT4 --version sci
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset WikiTopicsMT4 --version health
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset Metafam --version None
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset FBNELL --version None
# transductive
python3 src_relation/run.py --config src_relation/config/zeroshot_transductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset WDsinger --version None
python3 src_relation/run.py --config src_relation/config/zeroshot_transductive/n16_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset CoDExSmall --version None

# n = 32, ensemble = 16
# inductive entity
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n32_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset HM --version 3k
# inductive entity-relation
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n32_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset WikiTopicsMT2 --version org
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n32_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset WikiTopicsMT3 --version art
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n32_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset WikiTopicsMT3 --version infra
# transductive
python3 src_relation/run.py --config src_relation/config/zeroshot_transductive/n32_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset NELL23k --version None
python3 src_relation/run.py --config src_relation/config/zeroshot_transductive/n32_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset FB15k237_10 --version None

# n = 64, ensemble = 16
# inductive entity
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n64_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset ILPC2022 --version large
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n64_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset HM --version 5k
# transductive
python3 src_relation/run.py --config src_relation/config/zeroshot_transductive/n64_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset FB15k237_20 --version None
python3 src_relation/run.py --config src_relation/config/zeroshot_transductive/n64_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset FB15k237_50 --version None

# n = 128, ensemble = 16
# inductive entity
python3 src_relation/run.py --config src_relation/config/zeroshot_inductive/n128_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset HM --version indigo
# transductive
python3 src_relation/run.py --config src_relation/config/zeroshot_transductive/n128_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset NELL995 --version None
python3 src_relation/run.py --config src_relation/config/zeroshot_transductive/n128_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset ConceptNet100k --version None
python3 src_relation/run.py --config src_relation/config/zeroshot_transductive/n128_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset WN18RR --version None
python3 src_relation/run.py --config src_relation/config/zeroshot_transductive/n128_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset CoDExMedium --version None
python3 src_relation/run.py --config src_relation/config/zeroshot_transductive/n128_ensemble16.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset FB15k237 --version None

# n = 256, ensemble = 4
# transductive
python3 src_relation/run.py --config src_relation/config/zeroshot_transductive/n256_ensemble4.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset AristoV4 --version None

# n = 512, ensemble = 3
# transductive
python3 src_relation/run.py --config src_relation/config/zeroshot_transductive/n512_ensemble3.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset YAGO310 --version None

# n = 512, ensemble = 2
# transductive
python3 src_relation/run.py --config src_relation/config/zeroshot_transductive/n512_ensemble2_b4.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset CoDExLarge --version None
python3 src_relation/run.py --config src_relation/config/zeroshot_transductive/n512_ensemble2_b4.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset DBpedia100k --version None
python3 src_relation/run.py --config src_relation/config/zeroshot_transductive/n512_ensemble2_b4.yaml --gpus [0] --epochs 1 --bpe 1000 --ckpt $CKPT --dataset Hetionet --version None
