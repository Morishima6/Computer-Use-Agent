# 没有训练数据的情况下，训练k
train_k.py
    --generate-cases

输出：data/k.json

# 已经有训练数据的情况下，训练k
train_k.py

输出：data/k.json

# 已经有训练数据的情况下，再次生成训练数据，训练k
train_k.py
    --overwrite-cases
    
# 对K不满意，测试变化的k对 F1-Recall-Precision 的影响
test_k.py --k 0.9

输出：data/k_eval.json


## train_k.py 其他参数：
    --report-root：报告数据根目录（默认 Ambler-Agent/trajectory/trajectory_base）
    --cases-root：train cases 根目录（默认 Ambler-Agent/trajectory/retrieval/step_retrieval/train_k-data）
	    --limit-steps：最多处理多少个 step
	    --limit-pairs：训练时最多用多少对样本
	    --target-precision：训练时 尽量满足的precision门槛（代码会优先挑满足该门槛且 recall 更高的阈值，否则退回 F0.5 最优）
	    --out：训练结果写到哪里（默认 k.json）
	    --embedding-model：用于计算 train cases 的 embedding 的模型（默认 text-embedding-v4）
	    （默认行为）如果发现 train_case.json 缺 embedding，会自动现算并写回文件（写入到 embeddings[--embedding-model]；只需跑一次，后续 train/test 就不会重复算）
	    --only-materialize-embeddings：只把缺失的 embedding 写回 train_case.json，然后退出（不训练 k）
	    --eval：训练后立刻调用评测（等价于跑一次 test_k.py）
	    --eval-test-ratio / --eval-seed / --eval-out：评测用的抽样比例/种子/输出路径

## test_k.py 其他参数
	    --k：直接指定 k
	    --k-file：从 json 里读 k（默认 k.json）
	    --embedding-model：用于计算 train cases 的 embedding 的模型（默认 text-embedding-v4；如果 k.json 里带 embedding_model，会自动跟随）
	    （默认行为）评测时如果发现 train_case.json 缺 embedding，会自动现算并写回文件
	    --report-root / --cases-root：同上
	    --test-ratio：评测抽样比例（用 task_id:step_id hash 做稳定切分）
	    --seed：切分随机种子
	    --limit-steps：最多评测多少个 step
	    --out：评测结果输出（默认 k_eval.json）
