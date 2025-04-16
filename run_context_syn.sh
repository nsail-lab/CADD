DATA="./configs/context_jdd_synthetic" # base_jdd_synthetic base_in_the_wild
DATASET="jdd_synthetic_context" # "jdd_synthetic" "in_the_wild"
EPOCH=30
for TYPE in "c" # "c" "t" "ct"
do
  for MODEL in "lfcc_mfcc_lcnn" "lfcc_mfcc_mesonet" "lfcc_mfcc_specrnet" "whisper_lfcc_mfcc_lcnn" "whisper_lfcc_mfcc_mesonet" "whisper_lfcc_mfcc_specrnet" # "rawnet3" "lfcc_lcnn" "lfcc_mesonet" "lfcc_specrnet" "mfcc_lcnn" "mfcc_mesonet" "mfcc_specrnet" "whisper_lcnn" "whisper_mesonet" "whisper_specrnet" "whisper_lfcc_lcnn" "whisper_lfcc_mesonet" "whisper_lfcc_specrnet" "whisper_mfcc_lcnn" "whisper_mfcc_mesonet" "whisper_mfcc_specrnet"
  do
      for SEED in 0 1 2 # 1 2
      do
          CONFIG_F="${DATA}/${MODEL}_context.yaml"
          MODEL_PATH="./models/${MODEL}_context_context_${TYPE}_seed_${SEED}_epoch_${EPOCH}_train_${DATASET}"
          EMBEDDING_PATH="/tank/local/cgo5577/jdd_context-main/processed_embeddings/jdd-SYN_${TYPE}"
          python train_model.py \
               --config ${CONFIG_F} \
               --seed ${SEED} \
               --embedding_path ${EMBEDDING_PATH}

          python evaluate_context.py \
               --model-path ${MODEL_PATH} \
               --dataset ${DATASET} \
               --embedding_path ${EMBEDDING_PATH}
      done
  done
done
