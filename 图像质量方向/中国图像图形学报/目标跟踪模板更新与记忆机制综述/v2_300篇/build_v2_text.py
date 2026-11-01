from pathlib import Path
from collections import Counter,defaultdict
import re,json,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parent
STEM='01_综述完整初稿_v2_300篇文献_20261025'
records=json.loads((ROOT/'library_300.json').read_text(encoding='utf-8'))
bykey={r['key']:r for r in records}
def nums(values):
    a=sorted(set(values));spans=[];i=0
    while i<len(a):
        j=i
        while j+1<len(a) and a[j+1]==a[j]+1:j+=1
        if j-i>=2:spans.append(f'{a[i]}–{a[j]}')
        else:spans.extend(str(v) for v in a[i:j+1])
        i=j+1
    return '['+','.join(spans)+']'
def cite(*keys):return nums(bykey[k]['number'] for k in keys)
def resolve(s):
    return re.sub(r'\[@([^\]]+)\]',lambda m:cite(*m[1].split(',')),s)

sections={}
ext=(ROOT/'expanded_sections.md').read_text(encoding='utf-8')
for part in re.split(r'<!-- INSERT:([A-Z0-9]+) -->',ext)[1:][::2]:pass
chunks=re.split(r'<!-- INSERT:([A-Z0-9]+) -->',ext)
for i in range(1,len(chunks),2):sections[chunks[i]]=resolve(chunks[i+1].strip())+'\n\n'
text=(ROOT.parent/'03_综述完整初稿_v1_20261025.md').read_text(encoding='utf-8')
body,oldrefs=text.split('## 参考文献',1)
appendix='## 附录：数值来源与证据边界'+oldrefs.split('## 附录：数值来源与证据边界',1)[1]
body=body.replace('阅读初稿 v1.0｜2026年10月25日','文献扩展阅读稿 v2.0｜2026年10月25日｜300条参考文献')
body=body.replace('作者、单位与基金信息由实际作者确定，未在此代填。','本版将参考文献扩展为300条，并补充机制分析与主题索引；书目核验、全文精读和实验复现分别记录。作者、单位与基金信息由实际作者确定。')
body=body.replace('## 1 问题定义与统一分析框架',sections['INTRO']+'## 1 问题定义与统一分析框架')
body=body.replace('**表2 更新与记忆研究中的概念边界**',sections['FOUNDATION']+'**表2 更新与记忆研究中的概念边界**')
body=body.replace('## 3 时序记忆机制',sections['CHAPTER2']+'## 3 时序记忆机制')
body=body.replace('## 4 数据集、评价协议与文献结果分析',sections['CHAPTER3']+'## 4 数据集、评价协议与文献结果分析')
body=body.replace('## 6 结束语',sections['CHAPTER5']+'## 6 结束语')

cats=defaultdict(list)
for r in records:cats[r['category']].append(r)
inside=sum(1 for r in records if r.get('source_index') and ('thecvf.com' in r['source_index'] or 'ecva.net' in r['source_index']))
coverage=f'''### 4.6 扩展文献覆盖及可追溯性

本次扩展以官方会议目录、期刊原始页面和出版机构登记的DOI元数据为主要入口。CVF和ECVA目录的定向检索产生1016条去重候选记录，题名筛选后结合摘要核对任务范围与机制相关性，其中{inside}条进入本稿；其余{300-inside}条来自经典文献、期刊及其他会议来源。相同工作的预印本与正式发表版本只计一条，优先采用正式版本。目录记录、来源地址和分类结果保存在配套索引中。这一过程属于围绕主题的定向整理，不能作为覆盖全部期刊、数据库和语种的系统综述检索结果。

新增的249条CVF/ECVA记录均获取了官方摘要和书目信息，另补充经典研究、中文期刊及NeurIPS、ICLR文献。摘要核对用于任务筛选和初步机制定位，涉及细节实现、具体公式和实验解释时仍需回到原文相应章节。表9给出全部300条文献的主要组织位置，每条只按主要用途归入一组，以避免统计重复；一种方法可能同时具有多个机制属性。

**表9 本稿300条文献的主题组织与覆盖范围**

| 主题 | 篇数 | 在本稿中的用途 | 文献索引 |
|---|---|---|---|
'''
uses={
'更新策略与在线适应':'区分参数、模板和控制器更新',
'相关综述与研究定位':'定位本文相对已有综述的组织角度',
'相关滤波与样本正则化':'分析历史样本、可靠性与优化约束',
'孪生匹配与目标条件化':'建立匹配骨架与更新收益的对照',
'长时跟踪与记忆恢复':'分析回退、重检测与干扰候选',
'注意力与时序状态建模':'比较历史检索和跨帧状态表示',
'数据集与评价设计':'限定任务协议与证据可比范围',
'传统外观建模与局部可靠性':'追溯局部可靠性及增量建模基础',
'视频分割中的可迁移记忆机制':'借鉴记忆组织；保留分割协议边界',
'多模态历史与统一跟踪':'讨论模态质量、缺失与历史适配',
'高效跟踪与资源预算':'比较模型、token和历史存储成本',
'自回归与生成式跟踪':'分析状态递推与未来预测先验',
'自监督训练与表征迁移':'区分训练信号与推理时历史更新',
'无人机与退化观测':'提供延迟、遮挡和成像退化场景',
'语言条件与语义更新':'讨论语义历史与初始化协议差异',
'对抗扰动与可靠性诊断':'提出区分当前失败与后续污染的诊断'}
for cat,items in cats.items():coverage+=f"| {cat} | {len(items)} | {uses[cat]} | {nums(r['number'] for r in items)} |\n"
coverage+='\n表9中的文献索引用于指明该组研究的讨论入口，不表示每篇都采用相同的更新公式或均含独立记忆模块。分割、语言及多模态文献用于机制迁移和协议边界分析；本文不将这些任务的成绩混合为RGB单目标跟踪排行榜。\n\n'

fig=ROOT/'figures';fig.mkdir(exist_ok=True)
font=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'svg.fonttype':'none'})
years=Counter(r['year'] for r in records)
fig5,ax=plt.subplots(figsize=(11,3.5))
yy=sorted(years);xx=range(len(yy));bb=ax.bar(xx,[years[y] for y in yy],color='#2D728F',width=.72)
ax.set_xticks(list(xx),[str(y) for y in yy],rotation=45,ha='right');ax.set_ylabel('本稿选入篇数')
ax.set_title('300条文献的发表年份分布');ax.spines[['top','right']].set_visible(False)
ax.bar_label(bb,padding=3,fontsize=8);ax.set_ylim(0,max(years.values())+7)
fig5.tight_layout();fig5.savefig(fig/'fig5_years.png',dpi=220,bbox_inches='tight');fig5.savefig(fig/'fig5_years.svg',bbox_inches='tight');plt.close(fig5)
coverage+=f"![图5 本稿选入文献的年份分布]({(fig/'fig5_years.png').as_posix()})\n\n图5 本稿300条文献记录的年份分布。该图描述本稿选择的样本，不代表全领域发文量、引文影响或研究趋势强度。\n\n"
coverage+='''### 4.7 从方法比较到机制证据

表10将新增的部分关键工作放回统一框架。比较时首先确定状态载体和作用环节，再讨论它可能缓解的失效，最后回查是否存在相应消融或公平对照。这样可以减少仅按模型缩写、发表年份或总体分数排列方法所造成的误解。

**表10 新增代表工作的状态载体与分析位置**

| 工作 | 状态或历史载体 | 主要讨论环节 | 分析时的关键边界 |
|---|---|---|---|
'''
rows=[
('MEEM','C8','跟踪器历史快照','模型恢复','历史模型恢复不等于图像模板替换'),
('训练集自适应去污染','P147','样本与质量权重','样本管理','连续减权与硬阈值拒绝不同'),
('Meta-Updater','P389','几何、判别与外观序列线索','更新时机','决策何时更新，不直接等同于融合算子'),
('全局表示记忆读取','S2','查询相关的历史表示','历史选择和读取','读出的相关性与存储容量共同作用'),
('AQATrack','P741','初始模板与自回归查询','压缩状态传播','动态查询与显式历史图库不同'),
('RTracker','P782','正负样本树结构','跟踪检测协同恢复','须同时说明状态判断和搜索控制'),
('PURA','P869','测试时模型参数及更新方向','参数适应与恢复','作用对象为参数；输入为RGB-T'),
('XMem','P634','多时间尺度特征记忆','巩固与容量管理','分割任务的监督与输出不同'),
('SAM 2','C11','提示及预测相关的流式记忆','通用视频处理','需明确框、点、掩码及交互设置'),
('DAM4SAM','P833','干扰感知历史与更新判断','污染抑制','基于SAM2的记忆改造'),
('SAM2Long','P916','受限数量的候选路径','错误分支保留与选择','最终路径选择的时机影响因果协议'),
('DTPTrack','P944','校准后的历史与紧凑时序先验','可靠读取和状态引导','基础跟踪器与附加模块应分别分析'),
('TGTrack','P1002','历史条件与时序生成状态','时序监督','训练生成任务不等于访问真实未来帧'),
('UTPTrack','P1013','静态、动态模板及搜索token','统一压缩','压缩的不同信息来源需分别统计')]
for name,key,carrier,step,bound in rows:coverage+=f'| {name}{cite(key)} | {carrier} | {step} | {bound} |\n'
coverage+='\n表10是依据文献机制作出的归纳，不构成新增算法实验。对方法有效性的判断仍应与原论文的任务设置、基准和消融条件相联系；同一名称下不同输入分辨率、骨干或训练配置也应作为独立实验配置记录。\n\n'
body=body.replace('## 5 现存挑战与研究展望',coverage+'## 5 现存挑战与研究展望')

body=body.replace((ROOT.parent/'figures').as_posix(),fig.as_posix())
body=body.replace('最后，讨论更新价值估计、污染记忆修复、资源约束下的历史压缩以及可解释评测等问题','最后，结合在线适应、生成式时序建模和邻近任务的记忆设计，讨论更新价值估计、污染记忆修复、资源约束下的历史压缩以及可解释评测等问题')
refs='## 参考文献\n\n本阅读稿保留v1的文献[1–35]编号，并新增[36–300]以便逐条对照。所有条目均提供原始来源；同一工作的预印本与正式版本不重复计数。正式投稿时应按目标期刊要求统一编号、作者著录及格式。\n\n'+(ROOT/'references_300.md').read_text(encoding='utf-8')+'\n'
note='''\n\n## 附录：文献索引与精读状态

配套Excel包括全部300篇、优先精读50篇及分类核验说明。每条记录列出题名、作者、年份、出版出处、DOI（已取得者）、来源链接、主题、适用范围、正文位置及建议精读问题。P1、P2、P3为本稿建议的阅读顺序，不是论文质量排名。BibTeX和RIS文件用于文献管理软件导入。

官方元数据与摘要核对用于确认论文身份和主题，不能替代对完整方法、实现细节和实验条件的精读。索引因此分别保留核验层级与精读状态，不把自动获取记录标记为已全文阅读。本文未新增统一复现实验；原论文报告值未由本稿重新验证。
'''
full=body+refs+appendix+note
(ROOT/(STEM+'.md')).write_text(full,encoding='utf-8')
body_hanzi=len(re.findall(r'[\u4e00-\u9fff]',body))
print(json.dumps({'body_hanzi':body_hanzi,'references':len(records),'pool_selected':inside,'other_sources':300-inside,'tables':len(re.findall(r'^\*\*表\d+',body,re.M)),'figures':len(re.findall(r'^!\[',body,re.M))},ensure_ascii=False))
