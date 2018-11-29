mkdir -p result
mkdir -p figs

export CUDA_VISIBLE_DEVICES=1
for m in 1 0.75 0.5 0.25 0.1 0.01 0.001 0; do
    python -u src/run.py --data_dir data --result_dir result --dataset kdd --momentum $m
done

#python -u src/run.py --data_dir data --result_dir result --dataset kdd --momentum 0.5
