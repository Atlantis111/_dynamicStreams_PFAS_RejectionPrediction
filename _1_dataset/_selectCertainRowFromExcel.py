import pandas as pd


def extract_and_save_references(excel_file_path, output_txt_path='unique_references.txt'):
    """
    从Excel文件中提取reference列内容，去重后保存到txt文件

    参数:
        excel_file_path (str): Excel文件路径
        output_txt_path (str): 输出txt文件路径，默认为'unique_references.txt'
    """
    try:
        # 读取Excel文件
        print("正在读取Excel文件...")
        df = pd.read_excel(excel_file_path)

        # 检查reference列是否存在
        if 'reference' not in df.columns:
            # 尝试查找包含'reference'的列（不区分大小写）
            reference_columns = [col for col in df.columns if 'Typeys of contaminants' in col.lower()]
            if reference_columns:
                reference_col = reference_columns[0]
                print(f"注意：使用列 '{reference_col}' 代替 'reference'")
            else:
                print("可用的列名:")
                for i, col in enumerate(df.columns):
                    print(f"{i + 1}. {col}")
                raise ValueError("未找到'reference'列，请检查列名")
        else:
            reference_col = 'Types of contaminants'

        # 提取reference列
        references = df[reference_col]

        # 显示提取的基本信息
        print(f"提取前数据总量: {len(references)} 行")
        print(f"非空值数量: {references.count()} 行")

        # 去除空值
        references = references.dropna()
        print(f"去除空值后数据量: {len(references)} 行")

        # 去重处理
        unique_references = references.drop_duplicates()
        print(f"去重后唯一引用数量: {len(unique_references)} 条")
        print(f"删除了 {len(references) - len(unique_references)} 个重复项")

        # 将数据转换为字符串列表（确保格式正确）
        reference_list = unique_references.astype(str).tolist()

        # 保存到txt文件
        print(f"正在保存到 {output_txt_path}...")
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            for i, ref in enumerate(reference_list, 1):
                f.write(f"{ref}\n")

        print(f"✅ 成功保存 {len(reference_list)} 个唯一引用到 {output_txt_path}")

        # 显示前几个结果作为预览
        print("\n前5个唯一引用预览:")
        for i, ref in enumerate(reference_list[:5], 1):
            print(f"{i}. {ref}")

        return reference_list

    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 '{excel_file_path}'")
        return []
    except Exception as e:
        print(f"❌ 处理文件时出错: {str(e)}")
        return []


def advanced_extraction_with_options(excel_file_path, output_txt_path=None,
                                     sheet_name=0, custom_column=None,
                                     preserve_order=False):
    """
    高级版本的提取函数，支持更多选项

    参数:
        excel_file_path (str): Excel文件路径
        output_txt_path (str): 输出文件路径
        sheet_name (str/int): 工作表名称或索引
        custom_column (str): 自定义列名
        preserve_order (bool): 是否保持原始顺序
    """
    try:
        # 设置默认输出文件名
        if output_txt_path is None:
            output_txt_path = 'unique_references_advanced.txt'

        # 读取Excel文件
        df = pd.read_excel(excel_file_path, sheet_name=sheet_name)

        # 确定目标列
        target_column = custom_column if custom_column else 'Types of contaminants'

        if target_column not in df.columns:
            available_columns = ", ".join(df.columns)
            raise ValueError(f"列 '{target_column}' 不存在。可用列: {available_columns}")

        # 提取并处理数据
        references = df[target_column].dropna().astype(str)

        # 去重（可选择是否保持顺序）
        if preserve_order:
            # 保持第一次出现的顺序
            unique_references = references[~references.duplicated()]
        else:
            # 简单去重
            unique_references = references.drop_duplicates()

        # 保存结果
        unique_references.to_csv(output_txt_path, index=False, header=False, encoding='utf-8')

        print(f"高级模式：成功保存 {len(unique_references)} y个唯一引用到 {output_txt_path}")
        return unique_references.tolist()

    except Exception as e:
        print(f"高级模式处理出错: {e}")
        return []


# 主程序
if __name__ == "__main__":
    print("=== Excel Reference列提取工具 ===\n")

    # 设置文件路径（请修改为您的实际文件路径）
    excel_file = "allPoly.xlsx"  # 替换为您的文件路径

    # 方法1：基本提取模式
    print("方法1：基本提取模式")
    results = extract_and_save_references(excel_file)

    # 方法2：高级提取模式（可选）y
    print("\n" + "=" * 50)
    use_advanced = input("是否使用高级模式？(y/n): ").lower().strip()

    if use_advanced == 'y':
        print("\n方法2：高级提取模式")
        sheet_name = input("输入工作表名称或索引（直接回车使用默认）: ").strip()
        column_name = input("输入列名（直接回车使用'reference'）: ").strip() or 'reference'

        # 处理工作表名称
        if sheet_name and sheet_name.isdigit():
            sheet_name = int(sheet_name)
        elif not sheet_name:
            sheet_name = 0

        advanced_results = advanced_extraction_with_options(
            excel_file,
            output_txt_path='unique_references_advanced.txt',
            sheet_name=sheet_name,
            custom_column=column_name,
            preserve_order=True
        )

    print("\n程序执行完成！")