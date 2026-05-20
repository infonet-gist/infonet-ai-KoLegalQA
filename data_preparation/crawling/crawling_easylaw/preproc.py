# =================================================
# Text normalization rules for training data
# =================================================
# 1. Remove common footer in answers '\n저장\n페이스북\n트위터\n카카오톡'
#
# 2. Standardize formatting for training consistency
# - Normalize list markers such as '\n◇', '\n☞' into unified format
# - Use consistent bullet order: '-', '*', '+', (ensure a leading space before list markers for structure clarity)
# - Convert numbered circle bullets ① --> (1)
# - Normalize quotation marks: “ ” --> " "
# - Preserve legal quotation marks: Keep 「」 as-is for legal terms 
#   (optional: wrap legal terms like <LAW>민법</LAW>)
# - Normalize middle dots: ㆍ(아래아 - Highly likely to be treated as noise) --> ·(ASCII middle dot)
# - Normalize arrows: '->' --> '→'
# - Normalize multiplication symbols: X, x, ⅹ -> ×
# - Normalize units: e.g. '㎡', 'ℓ', '℃' --> 'm^2', 'L', 'C'
# - Note on '※': Keep as-is (models can interpret)
#
# 3. Remove broken table-like structures
#    (discarded due to garbled or misaligned tabular text)
#=================================================#

import os
import json
import re
from collections import deque

def infer_hierarchy(text):
    stack = deque()
    lines = text.strip().split('\n')
    result = []
    
    # number pattern 1.
    number_dot_pattern = re.compile(r'^(\d+\.\s)')
    # number pattern (1)
    number_paren_pattern = re.compile(r'^(\(\d+\))')
    # symbol pattern
    symbol_markers = '◇☞▪•·⦁・√-⁃□￭◈'
    symbol_pattern = re.compile(r'^([' + re.escape(symbol_markers) + '])')
    # output bullet pattern
    bullet_order = ['-', '*', '+']

    for line in lines:
        line = line.strip()
        if not line:
            continue

        marker = ''
        content = line

        m_num_dot = number_dot_pattern.match(line)
        m_num_par = number_paren_pattern.match(line)
        m_sym = symbol_pattern.match(line)

        if m_num_dot:
            marker = m_num_dot.group(1).strip()
            content = line[m_num_dot.end():].strip()
            marker_type = 'num_dot'
        elif m_num_par:
            marker = m_num_par.group(1).strip()
            content = line[m_num_par.end():].strip()
            marker_type = 'num_par'
        elif m_sym:
            marker = m_sym.group(1)
            content = line[m_sym.end():].strip()
            marker_type = 'symbol'
        else:
            marker_type = 'text'

        if marker_type == 'symbol':
            if marker in stack:
                while stack and stack[-1] != marker:
                    stack.pop()
            else:
                stack.append(marker)

        elif marker_type == 'num_dot' or marker_type == 'num_par':
            if marker_type in stack:
                while stack and stack[-1] != marker_type:
                    stack.pop()
            else:
                stack.append(marker_type)

        level = len(stack) - 1
        
        #print(stack)

        if marker_type == 'symbol':
            bullet = bullet_order[level % len(bullet_order)]
        else:
            bullet = marker

        # output line
        if marker_type =='text':
            result.append(f"{content}")
        else:
            indent = '  ' * level
            result.append(f"{indent}{bullet} {content}")

    return '\n'.join(result)

def remove_common_footer(text):
    footer_pattern = '\n저장\n페이스북\n트위터\n카카오톡'
    text = re.sub(footer_pattern, '', text)
    return text

def remove_unicode_symbols(text):
    circle_characters = {
        '①': '(1)',
        'ⓛ': '(1)',
        '②': '(2)',
        '③': '(3)',
        '④': '(4)',
        '⑤': '(5)',
        '⑥': '(6)',
        '⑦': '(7)',
        '⑧': '(8)',
        '⑨': '(9)',
        '⑩': '(10)',
        '⑪': '(11)',
        '⑫': '(12)',
        '⑬': '(13)',
        '⑭': '(14)',
        '⑮': '(15)',
        '㉠': '(1)',
        '㉡': '(2)', 
        '㉢': '(3)', 
        '㉣': '(4)',
    }
    
    quote_marks = {
        '“':  '"',
        '”':  '"',
        '‘':  "'",
        '’':  "'",
        '『': '"',
        '』': '"',
        '「': '<LAW>',
        '」': '</LAW>',
        '｢': '<LAW>',
        '｣': '</LAW>',
        '＜': '<',
        '＞': '>',
        'ʻ': "'",
        'ʼ': "'",
        '´': "'",
    }
    
    etc = {
        ' ': ' ',
        '☎ ': '',
        '☎': '',
        '☏ ': '',
        '☏': '',
        '∼': '~',
        '～': '~',
        '↔': '~',
        '△': '-',   # marker used in tables (represent mid of O/X)
        '▶': '→',
        '->': '→',
        '［': '[',
        '］': ']',
        '｛': '{',
        '｝': '}',
        ' · ': '·',
        'ㆍ': '·',
        'ᆞ': '·',
        '‧': '·',
        '・': '·',
        '∙': '·',
        '･': '·',
        '‹': '·',
        '：': ':',
        '±': '+/-', # appears only once in text (±5 kWh) may not parse correctly
        '÷': '/',
        '％': '%',
        '‑': '-',
        '–': '-',
        '－': '-',
        '―': '-',
        'ㅡ': '-',
        '…': '...',
        'Ⅰ': 'I',   # Roman num
        'Ⅱ': 'II',
    }
    
    for symbol in circle_characters:
        text = re.sub(symbol, circle_characters[symbol], text)
    
    for mark in quote_marks:
        text = re.sub(mark, quote_marks[mark], text)
        
    for chr in etc:
        text = re.sub(chr, etc[chr], text)
    
    text = re.sub('◇\n', '◇', text)
    text = re.sub('☞\n', '☞', text)
    text = re.sub(' : ', ':', text)
    text = re.sub(' ~ ', '~', text)
    text = re.sub('ΟΟ', 'ㅇㅇ', text)
    text = re.sub('CO₂', 'CO2', text)
    text = re.sub(' √ ', ' ', text) # case: 창업/관건물위생관리업을 하는 사람
    text = re.sub('\n﻿', '', text)  # u+feff
    text = re.sub('\n', '', text)  # u+f0df
    text = re.sub('\n', '', text)  # u+f0e0
    text = re.sub('\n', '', text)  # u+f0e2
        
    return text

def fix_metric(text):
    metric_list = {
        '℃': '°C',
        'ℓ': 'L',
        '㎍': 'μg',
        '㎎': 'mg',
        '㎖': 'mL',
        '㎛': 'μm',
        '㎝': 'cm',
        '㎞': 'km',
        '㎟': 'mm^2',
        '㎡': 'm^2',
        '㎢': 'km^2',
        '㎤': 'cm^3',
        '㎥': 'm^3',
        '㏃': 'Bq',
        '㏄': 'cc',
    }
    
    for metric in metric_list:
        text = re.sub(metric, metric_list[metric], text)
        
    return text
    
# Test Data
#example_text = "일반문장1\n◇ 주제1\n☞ 설명1\n1. 숫자1\n2. 숫자2\n☞ 설명2\n☞ 설명3\n◇ 주제2\n☞ 설명4\n일반문장2"
#example_text = "고소, 고발, 자수, 신고, 인지 등으로 수사기관은 수사를 시작하며, 범죄의 혐의가 있다고 생각되면 범죄사실과 증거에 대해 수사합니다.\n수사기관이 수사를 하는 경우 피의자는 상황에 따라 체포되거나 구속될 수 있으며, 수사가 완료되어 혐의가 인정되면 검사는 공소를 제기합니다.\n1. 시작\n☞ 고소 : 피해자나 피해자의 법정대리인과 같은 고소권자가 경찰이나 검찰에 고소를 하면 수사가 시작됩니다.\n☞ 고발 : 고소권자가 아니더라도 범죄행위에 대해 누군가가 경찰이나 검찰에 고발을 하면 수사가 시작됩니다.\n☞ 신고 : 누군가가 범죄행위에 대해 경찰에 신고를 한 경우 경찰은 상황을 파악한 후 수사를 시작합니다.\n☞ 인지 : 고소, 고발, 신고 등이 없더라도 범죄행위를 경찰이나 검찰이 스스로 인지한 경우에도 수사를 시작할 수 있습니다.\n2. 체포·구속\n① 피의자(가해자)가 수사기관의 출석요구에 응하지 않거나,\n② 주거가 일정하지 않거나,\n③ 증거를 인멸할 우려가 있거나,\n④ 도주 우려 등이 있는 경우에는 검사가 판사에게 체포·구속영장을 청구하여 이를 발부받은 후 체포·구속할 수 있습니다.\n3. 수사(경찰)\n☞ 경찰은 피의자신문, 증거조사 등의 방법으로 수사를 진행합니다.\n4. 사건을 검찰로 송치\n☞ 경찰은 수사가 종료되면 해당 사건을 검찰로 송치합니다.\n5. 수사(검찰)\n☞ 검찰은 경찰이 보낸 수사자료를 바탕으로 수사를 진행합니다.\n6. 공소제기(기소), 불기소처분\n☞ 경찰로부터 송치 받은 사건이나 검찰 스스로 수사를 한 사건이 종결되면 검찰은 법원에 공소를 제기하거나 불기소처분을 합니다.\n저장\n페이스북\n트위터\n카카오톡"

def save_merged():
    categories = [
        '가정법률',
        '교통_운전',
        '국가_지자체',
        '국방_보훈',
        '근로_노동',
        '금융_금전',
        '무역_출입국',
        '문화_여가생활',
        '민형사_소송',
        '복지',
        '부동산_임대차',
        '사업',
        '사회안전_범죄',
        '소비자',
        '아동청소년_교육',
        '정보통신_기술',
        '창업',
        '환경_에너지',
    ]

    merged_data =[]
    for category in categories:
        input_file = 'crawling_output/'+category+'.json'
        with open(input_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        for item in json_data:
            title = item.get('title', '').strip()
            query = remove_unicode_symbols(title)
            
            content = item.get('content','').strip()
            response = remove_common_footer(content)
            response = remove_unicode_symbols(response)
            response = fix_metric(response)
            response = infer_hierarchy(response)
            
            processed_item = {
                'query': query,
                'response': response,
                'category': category
            }
            merged_data.append(processed_item)
        
        print(f'{category} preprocessing completed')
        
    output_dir = 'preproc_output'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir,'preproc_easylaw.json')

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
        
    print("File saved: {output_file}")
    
    
def view_json_one_by_one(file_path: str):
    
    def load_file(path: str):
        with open(path, 'r', encoding='utf-8') as f:
            dat = json.load(f)
        return dat

    data = load_file(file_path)
    print(f"Total {len(data)} items.\n")

    idx = 0
    while True:
        item = data[idx]
        print(f"\n--- [{idx}/{len(data)}] ---")
        print(f"Query:\n{item.get('query', '')}\n")
        print(f"Answer:\n{item.get('response', '')}\n")
        print(f"Category: {item.get('category', '')}\n")
        
        keyin = input("...")
        
        if keyin.lower() in ['x', 'end', 'exit', 'finish', 'quit', 'stop']:
            print("Exiting file viewer.")
            break
        elif keyin.lower() in ['r', 'refresh', 'reload']:
            data = load_file(file_path)
        elif keyin.isdigit():
            idx = int(keyin) % len(data)
        elif keyin.startswith('='):
            try:
                idx = int(keyin[1:]) % len(data)
            except ValueError:
                idx = (idx + 1) % len(data)
        elif keyin.startswith('+') or keyin.startswith('-'):
            try:
                idx = (idx + int(keyin)) % len(data)
            except ValueError:
                idx = (idx + 1) % len(data)
        else:
            idx = (idx + 1) % len(data)
                
            
def collect_used_characters(file_path: str):
    if not os.path.exists(file_path):
        print(f"no file: {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    character_set = set()

    for item in data:
        response_text = item.get('response', '')
        character_set.update(response_text)

    # remove dupes and sort
    character_list = sorted(character_set)

    return character_list

def find_wierd_unicode(file_path: str):
    target_chars = ['\uf0df', '\uf0e0', '\uf0e2', '\ufeff']
    context_window = 30
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    results = []

    for chr in target_chars:
        idx = 0
        while True:
            idx = text.find(chr, idx)
            if idx == -1:
                break
            start = max(0, idx - context_window)
            end = min(len(text), idx + 1 + context_window)
            snippet = text[start:end]
            results.append({
                'character': f'U+{ord(chr):04X}',
                'position': idx,
                'context': snippet
            })
            idx += 1

    return results


def put_num(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for i, item in enumerate(data):
        item['num'] = str(i)
    
    out_file = 'preproc_easylaw_numbers.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
def delete_by_num(file_path: str):
    target_nums = [
    '10', '100', '107', '108', '137', '145', '153', '154', '159', '167', '168', '181', '190', '214', '244', '249', '255', '256', '271', '283', '302', '305',
    '312', '322', '323', '327', '329', '335', '336', '337', '339', '342', '345', '346', '348', '350', '351', '352', '359', '418', '419', '422', '436', '437',
    '438', '441', '443', '444', '451', '453', '454', '462', '467', '468', '469', '470', '471', '472', '474', '504', '510', '538', '543', '544', '550', '619',
    '625', '629', '652', '655', '660', '662', '669', '679', '683', '745', '761', '765', '777', '792', '801', '803', '805', '808', '820', '822', '823', '825',
    '843', '851', '870', '871', '882', '887', '932', '937', '940', '956', '959', '974', '980', '987', '994', '1009', '1031', '1038', '1054', '1056', '1057',
    '1071', '1077', '1079', '1080', '1096', '1097', '1098', '1106', '1107', '1108', '1112', '1123', '1127', '1130', '1132', '1133', '1136', '1143',
    '1144', '1146', '1148', '1152', '1156', '1158', '1159', '1163', '1164', '1167', '1185', '1208', '1263', '1264', '1278', '1280', '1313', '1314',
    '1343', '1344', '1345', '1348', '1362', '1364', '1373', '1374', '1375', '1377', '1378', '1389', '1400', '1402', '1404', '1408', '1419', '1420',
    '1421', '1422', '1424', '1426', '1427', '1429', '1441', '1442', '1444', '1476', '1482', '1504', '1542', '1544', '1555', '1561', '1627', '1628',
    '1647', '1669', '1708', '1713', '1715', '1717', '1718', '1725', '1726', '1732', '1737', '1738', '1747', '1773', '1775', '1784', '1786', '1834',
    '1849', '1851', '1864', '1865', '1867', '1875', '1880', '1887', '1888', '1904', '1906', '1907', '1935', '1949', '1981', '1988', '2024', '2037',
    '2058', '2059', '2060', '2062', '2088', '2152', '2162', '2200', '2208', '2209', '2256', '2260', '2266', '2284', '2294', '2295', '2307', '2311',
    '2317', '2335', '2338', '2388', '2394', '2396', '2416',
    '253', '895', '917', '1006', '1011', '1035', '1037', '1745', '1748', '2143', '2168',
    '1598',
    ]
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    filtered_data = [item for item in data if item.get("num") not in target_nums]

    for item in filtered_data:
        item.pop("num", None)
        
    out_file = 'preproc_easylaw_cleanup.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        
        

output_file = 'preproc_output/preproc_easylaw.json'
#save_merged()
#view_json_one_by_one(output_file)
#print(collect_used_characters(output_file))
#print(find_wierd_unicode(output_file))
#put_num('preproc_output/preproc_easylaw.json')
#delete_by_num('preproc_output/preproc_easylaw_numbers.json')