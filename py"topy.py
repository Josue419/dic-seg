import os

def fix_py_suffixes(root_dir):
    # 递归遍历所有目录和文件
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # 检查文件名是否包含 .py 且后面有多余字符（如 .py"）
            if '.py' in filename and not filename.endswith('.py'):
                # 分割出 .py 之前的部分，重新拼接正确的 .py 后缀
                # 例如 "test.py\" -> 分割为 "test" + ".py"
                base_name = filename.split('.py', 1)[0]
                new_filename = f"{base_name}.py"
                
                # 构建完整路径
                old_path = os.path.join(dirpath, filename)
                new_path = os.path.join(dirpath, new_filename)
                
                # 避免重命名到已存在的文件
                if not os.path.exists(new_path):
                    try:
                        os.rename(old_path, new_path)
                        print(f"已修正: {old_path} -> {new_path}")
                    except Exception as e:
                        print(f"修正失败 {old_path}: {e}")
                else:
                    print(f"跳过 {old_path}（目标文件 {new_filename} 已存在）")

if __name__ == "__main__":
    # 获取当前工作目录
    current_dir = os.getcwd()
    print(f"开始处理目录: {current_dir} 及其子目录...")
    fix_py_suffixes(current_dir)
    print("处理完成")