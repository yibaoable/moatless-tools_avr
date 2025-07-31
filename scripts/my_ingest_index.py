import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
import requests
import argparse  # 新增导入argparse模块
sys.path.append("..")
print(f"sys.path: {sys.path}")

import logging

from moatless.index import CodeIndex, IndexSettings
from moatless.repository import FileRepository

# 先测试API端点可达性
try:
    test_response = requests.get("https://ark.cn-beijing.volces.com/api/v3", timeout=10)
    print(f"API endpoint status: {test_response.status_code}")
except Exception as e:
    print(f"API connection test failed: {str(e)}")
    raise


# 1. 首先获取嵌入模型的实际维度
def get_embedding_dimension(model_name: str) -> int:
    """动态获取嵌入模型的输出维度"""
    if model_name.startswith("ep-"):  # 火山方舟模型
        # 这里假设使用OpenAIEmbedding封装
        from llama_index.embeddings.openai import OpenAIEmbedding
        # os.environ["ARK_API_KEY"] = "Bearer " + os.environ["ARK_API_KEY"]
        embed_model = OpenAIEmbedding(
            model_name=model_name,
            api_key=os.environ.get("ARK_API_KEY"),
            api_base="https://ark.cn-beijing.volces.com/api/v3",
            api_type = "ark",
            timeout=60,
        )
    elif model_name == "BAAI/bge-code-v1":
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        embed_model = HuggingFaceEmbedding(model_name=model_name)
    else:  # 其他模型
        raise ValueError(f"Unsupported model: {model_name}")
    
    # 获取测试文本的嵌入维度
    test_embedding = embed_model.get_text_embedding("test")
    return len(test_embedding)

def main():
    # 2. 设置命令行参数解析
    parser = argparse.ArgumentParser(description='Code Indexing Tool')
    parser.add_argument('--repo_dir', type=str, required=True, help='Path to the repository directory')
    parser.add_argument('--persist_dir', type=str, required=True, help='Path to store the index')
    args = parser.parse_args()

    # 3. 设置索引参数
    model_name = "ep-20250718205428-zr9hd"
    # actual_dim = get_embedding_dimension(model_name)  # 动态获取维度

    index_settings = IndexSettings(
        embed_model=model_name,
        dimensions=2560  # 使用实际维度而非硬编码值
    )

    print(f"Embedding model: {index_settings.embed_model}, actual dimensions: {index_settings.dimensions}")
    print(f"ARK_API_KEY: {os.environ.get('ARK_API_KEY')}")

    # 4. 初始化代码索引
    file_repo = FileRepository(repo_path=args.repo_dir)
    code_index = CodeIndex(file_repo=file_repo, settings=index_settings)

    # 5. 运行索引过程
    try:
        nodes, tokens = code_index.run_ingestion()
        print(f"Indexed {nodes} nodes and {tokens} tokens")
        code_index.persist(args.persist_dir)
    except Exception as e:
        print(f"Error during indexing: {str(e)}")
        raise

if __name__ == "__main__":
    main()