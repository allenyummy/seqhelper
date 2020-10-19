from tqdm import tqdm
from src.scheme import IOB2
from src.entity import EntityFromNestedList
input_nested_list = [[
                        ("台", "B-LOC"), 
                        ("北", "I-LOC"), 
                        ("是", "O"), 
                        ("阿", "B-PER"), 
                        ("倫", "I-PER"), 
                        ("的", "O"), 
                        ("家", "O")],
                     [
                        ("阿", "B-PER"),
                        ("倫", "I-PER"),
                        ("是", "O"),
                        ("人", "B-ANI")]
                    ]
df = EntityFromNestedList(input_nested_list, IOB2).chunks2df()
print (df)
#   pid type start_position end_position text
# 0   0  LOC              0            1   台北
# 1   0  PER              3            4   阿倫
# 2   1  PER              0            1   阿倫
# 3   1  ANI              3            3    人


# nested_list = list()
# temp_list = list()
# with open("train_1_update.txt", "r", encoding="utf-8") as f:
#     for line in tqdm(f):
#         line = line.rstrip()
#         if line == "":
#             nested_list.append(temp_list)
#             temp_list = list()
#         else:
#             split = line.split()
#             char = split[0]
#             label = split[1]
#             temp_list.append((char, label))

# print ("finish reading files")
# df = EntityFromNestedList(nested_list, IOB2).chunks2df()
# df.to_excel("output.xlsx", index=False)