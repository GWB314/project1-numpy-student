"""NumPy 数值计算练习 —— 12 道题的实现与运行结果

对应题目：practice/project1_exercises.pdf（8 道基础题 NP-E1~E8 + 4 道中等题 NP-M1~M4）
运行方式：python numpy_homework.py
只依赖 NumPy；每题的结果都由给定输入真实计算得出，没有写死的数值。
展示保留 6 位小数，核对允许绝对误差 1e-6。
"""

import numpy as np


# ---------------------------------------------------------------------------
# 通用输出工具
# ---------------------------------------------------------------------------
def fmt(x):
    """数组统一按 6 位小数打印，避免 1e-17 这类浮点噪声干扰阅读。"""
    arr = np.asarray(x, dtype=float)
    return np.array2string(arr, precision=6, suppress_small=False, separator=', ')


def f6(x):
    """标量按 6 位小数打印。"""
    return f'{float(x):.6f}'


def title(text):
    print(f'\n{"=" * 68}\n{text}\n{"=" * 68}')


# ---------------------------------------------------------------------------
# NP-E1 数组、形状与切片 [基础]
# ---------------------------------------------------------------------------
def np_e1():
    title('NP-E1 数组、形状与切片')

    A = np.array([[0.20, -0.50, 1.10, 0.40],
                  [0.80, 0.30, -0.70, 0.60],
                  [-0.40, 0.90, 0.10, -0.20]])

    # 1. 描述数组的三个属性
    print(f'1) ndim={A.ndim}  shape={A.shape}  size={A.size}')
    print('   说明：ndim 是轴的数量（2 维），shape 是每个轴的长度 (3,4)，'
          'size 是所有元素个数 3*4=12。')

    # 2. 第 2 行（下标 1）与最后 2 列
    row1 = A[1]            # 只给一个下标时，取的是整行，轴数从 2 降到 1
    last2 = A[:, -2:]      # 逗号左边是行轴（全要），右边是列轴（取最后 2 列）
    print(f'2) 第 2 行      : {fmt(row1)}   shape={row1.shape}')
    print(f'   最后 2 列     : {fmt(last2)}   shape={last2.shape}')
    print('   说明：A[1] 取一行得到 1 维 (4,)；A[:, -2:] 保留二维结构，'
          '所以是 (3,2)。切片用的是视图，并没有复制数据。')

    # 3. 复制后把负数改成 0，A 保持不变
    B = A.copy()           # 必须 copy，否则 B 和 A 指向同一块内存
    B[B < 0] = 0.0         # 布尔掩码：True 的位置被赋成 0
    print(f'3) B（负数已置 0）: {fmt(B)}')
    print(f'   A 未被修改      : {fmt(A)}')
    print('   说明：布尔索引 B[B<0]=0 会就地修改；因为先 copy 了一份，'
          'A 仍然是原值。')

    # 边界输入：3x4 之外的空维度会不会出错
    edge = np.zeros((0, 4))
    print(f'   边界输入 (0,4) 空数组: shape={edge.shape}, size={edge.size}')


# ---------------------------------------------------------------------------
# NP-E2 逐元素乘法、点乘与长度 [基础]
# ---------------------------------------------------------------------------
def np_e2():
    title('NP-E2 逐元素乘法、点乘与长度')

    a = np.array([0.80, -0.60, 0.70])
    b = np.array([-0.20, 0.90, 0.40])

    # 1. a*b 是逐元素，a@b 是求和归约
    elementwise = a * b
    dot = a @ b
    print(f'1) a * b = {fmt(elementwise)}   shape={elementwise.shape}')
    print(f'   a @ b = {f6(dot)}   shape={np.shape(dot)}（标量，没有轴）')
    print('   说明：* 对应位置相乘仍返回数组；@ 把所有乘积再加起来，'
          '得到一个数，所以 shape 是 ()。')

    # 2. 手工实现欧氏长度：平方 -> 求和 -> 开平方
    len_a = np.sqrt(np.sum(a ** 2))
    len_b = np.sqrt(np.sum(b ** 2))
    print(f'2) ||a|| = {f6(len_a)}   ||b|| = {f6(len_b)}')

    # 3. 用 np.linalg.norm 核对
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    print(f'3) np.linalg.norm: ||a|| = {f6(norm_a)}   ||b|| = {f6(norm_b)}')
    print(f'   两种算法是否一致（容差 1e-6）: '
          f'{np.allclose(len_a, norm_a, atol=1e-6) and np.allclose(len_b, norm_b, atol=1e-6)}')
    print('   说明：np.sum(a**2) 先把 3 个元素平方再沿唯一的轴求和，'
          '结构上等价于 a@a。')

    # 边界输入：全零向量的长度
    print(f'   边界输入 零向量: ||0|| = {f6(np.linalg.norm(np.zeros(3)))}')


# ---------------------------------------------------------------------------
# NP-E3 两个向量的夹角 [基础]
# ---------------------------------------------------------------------------
def angle_deg(a, b):
    """返回两个同维非零一维向量的夹角，单位为度。

    输入：a, b 为 1 维数组，shape 均为 (D,)
    输出：标量 float，单位度，范围 [0,180]
    异常：任一输入是零向量时抛 ValueError（角度没有定义）
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError('夹角无定义：输入包含零向量')

    # cos(theta) = (a @ b) / (||a|| * ||b||)
    cosine = float(a @ b) / (norm_a * norm_b)

    # 浮点误差可能让余弦略微超出 [-1,1]，clip 之后 arccos 才不会得到 nan
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def np_e3():
    title('NP-E3 两个向量的夹角')

    a = np.array([1.00, -1.00])
    b = np.array([1.00, 0.00])

    print(f'1) a @ b = {f6(a @ b)}')
    print(f'   ||a|| = {f6(np.linalg.norm(a))}   ||b|| = {f6(np.linalg.norm(b))}')
    print(f'   cos(theta) = {f6((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))}')

    print(f'2) 夹角 = {f6(angle_deg(a, b))} 度')
    print('   说明：cos 先 clip 到 [-1,1] 再 arccos，最后 np.degrees 把弧度转成度。')

    print('3) 边界输入（零向量）:')
    try:
        angle_deg(np.array([0.0, 0.0]), b)
    except ValueError as exc:
        print(f'   抛出 ValueError: {exc}')
    print('   说明：零向量没有方向，夹角无定义，所以显式报错而不是返回 nan。')


# ---------------------------------------------------------------------------
# NP-E4 两个点的距离 [基础]
# ---------------------------------------------------------------------------
def np_e4():
    title('NP-E4 两个点的距离')

    p = np.array([-0.50, 1.20, 2.00])
    q = np.array([0.80, -0.40, 1.10])

    # 1. 差向量
    diff = p - q
    print(f'1) p - q = {fmt(diff)}   shape={diff.shape}')
    print('   说明：两个 (3,) 相减仍是 (3,)，逐坐标做差。')

    # 2. 欧氏距离：平方和再开方（等价于差向量的长度）
    euclidean = np.sqrt(np.sum(diff ** 2))
    print(f'2) 欧氏距离 = {f6(euclidean)}')
    print(f'   用 np.linalg.norm 核对 = {f6(np.linalg.norm(diff))}')

    # 3. 曼哈顿距离：绝对值之和
    manhattan = np.sum(np.abs(diff))
    print(f'3) 曼哈顿距离 = {f6(manhattan)}')
    print('   说明：欧氏距离是 L2（先平方），曼哈顿距离是 L1（先取绝对值），'
          '后者恒不大于前者。')

    # 边界输入：同一点，距离应为 0
    print(f'   边界输入 两点重合: 欧氏 = {f6(np.linalg.norm(p - p))}, '
          f'曼哈顿 = {f6(np.sum(np.abs(p - p)))}')


# ---------------------------------------------------------------------------
# NP-E5 把向量变为单位长度 [基础]
# ---------------------------------------------------------------------------
def unit_vector(a):
    """返回方向相同、长度为 1 的新数组，不修改输入。

    输入：a 为 1 维数组，shape (D,)
    输出：新数组，shape (D,)，||输出|| = 1
    异常：输入是零向量时抛 ValueError
    """
    a = np.asarray(a, dtype=float)
    length = np.linalg.norm(a)
    if length == 0.0:
        raise ValueError('单位向量无定义：输入是零向量')
    return a / length      # 除法生成新数组，原数组 a 不受影响


def np_e5():
    title('NP-E5 把向量变为单位长度')

    a = np.array([-1.20, 0.40, 0.80])
    original = a.copy()

    # 1. 除以自身长度
    u = unit_vector(a)
    changed = not np.array_equal(a, original)
    print(f'1) 单位向量 = {fmt(u)}   shape={u.shape}')
    print(f'   原数组是否被改动: {"是" if changed else "否"}（当前内容 {fmt(a)}）')
    print('   说明：a / length 走的是返回新数组的路径，不是就地运算；'
          '若要就地改可以写 a /= length，那就会破坏原数据。')

    # 2. 零向量
    print('2) 边界输入（零向量）:')
    try:
        unit_vector(np.zeros(3))
    except ValueError as exc:
        print(f'   抛出 ValueError: {exc}')

    # 3. 核对长度
    length_u = np.linalg.norm(u)
    print(f'3) ||u|| = {f6(length_u)}')
    print(f'   np.allclose(||u||, 1) = {np.allclose(length_u, 1.0, atol=1e-6)}')


# ---------------------------------------------------------------------------
# NP-E6 沿不同 axis 汇总 [基础]
# ---------------------------------------------------------------------------
def np_e6():
    title('NP-E6 沿不同 axis 汇总')

    A = np.array([[0.20, -0.50, 1.10],
                  [0.80, 0.40, -0.30]])
    print(f'输入 A = {fmt(A)}   shape={A.shape}')

    # 1. axis=0：沿行方向压缩，每个位置留下“各行的和”
    col_sum = A.sum(axis=0)
    print(f'1) A.sum(axis=0)  = {fmt(col_sum)}   shape={col_sum.shape}')
    print('   说明：axis=0 表示把第 0 轴（行）消掉，剩下 3 列，'
          '每个值是同一列上 2 行的和。')

    # 2. axis=1：沿列方向压缩，每个位置留下“各行的平均”
    row_mean = A.mean(axis=1)
    print(f'2) A.mean(axis=1) = {fmt(row_mean)}   shape={row_mean.shape}')
    print('   说明：axis=1 把第 1 轴（列）消掉，剩下 2 行，'
          '每个值是同一行 3 个元素的平均。')

    # 3. 整体最大值
    pos = np.unravel_index(np.argmax(A), A.shape)
    print(f'3) A.max() = {f6(A.max())}，位于 ({int(pos[0])}, {int(pos[1])})（行, 列）')
    print('   说明：axis=0 的结果对应“竖着看”的三列之和（输入里上下两个数相加）；'
          'axis=1 的结果对应“横着看”的两行平均（输入里每一行内部平均）。')

    # 边界输入：单行数组，两种汇总的差别
    single = np.array([[1.0, 2.0, 3.0]])
    print(f'   边界输入 (1,3): axis=0 -> {fmt(single.sum(axis=0))} '
          f'(shape {single.sum(axis=0).shape}); '
          f'axis=1 -> {fmt(single.mean(axis=1))} (shape {single.mean(axis=1).shape})')


# ---------------------------------------------------------------------------
# NP-E7 一次广播，处理所有行 [基础]
# ---------------------------------------------------------------------------
def np_e7():
    title('NP-E7 一次广播，处理所有行')

    X = np.array([[0.80, -0.60, 0.70],
                  [-0.40, 0.50, 0.20]])
    scale = np.array([1.50, 0.50, -1.00])
    bias = np.array([-0.10, 0.20, 0.30])

    # 1. 一条表达式，不用循环
    Y = X * scale + bias
    print('1) Y = X * scale + bias')
    print(f'   Y = {fmt(Y)}   shape={Y.shape}')
    print(f'   X 的 shape={X.shape}，scale 的 shape={scale.shape}，'
          f'bias 的 shape={bias.shape}')

    # 2. 输出
    print(f'2) 输出 shape = {Y.shape}（与 X 相同）')

    # 3. 广播规则
    print('3) 说明：X 是 (2,3)，scale 和 bias 是 (3,)。')
    print('   NumPy 从最右边的轴开始对齐：(2,3) 与 (3,) 相比，'
          '长度 3 与 3 匹配，缺失的第 0 轴被看作 1，')
    print('   于是 scale 和 bias 相当于被“复制”成 (2,3)，'
          '每一行都用同一组 3 个缩放系数和偏置。')
    print('   这里第 0 轴（行）被广播，第 1 轴（特征）是一一对应的，'
          '所以只需一条表达式。')

    # 边界输入：只处理一行时结果相同
    print(f'   边界输入 只喂第一行: {fmt(X[:1] * scale + bias)} '
          f'（与上面第一行一致）')


# ---------------------------------------------------------------------------
# NP-E8 把两张图像组成一个 batch [基础]
# ---------------------------------------------------------------------------
def np_e8():
    title('NP-E8 把两张图像组成一个 batch')

    a = np.array([[[0.20, -0.40], [0.70, 0.10]],
                  [[0.30, 0.80], [-0.20, 0.50]],
                  [[-0.10, 0.60], [0.40, -0.30]]])
    b = a + 0.10

    print(f'单张图像 a 的 shape = {a.shape}   (C,H,W) = (3,2,2)')

    # 1. np.stack 新增第 0 轴作为图像编号
    batch = np.stack((a, b), axis=0)
    print(f'1) batch = np.stack((a,b))   shape={batch.shape}')
    print('   说明：stack 不是拼接（concatenate），它在指定位置新建一个轴，'
          '这里是第 0 轴，于是 (3,2,2) 变成 (2,3,2,2)。')

    # 2. 取蓝色通道（C=2）
    blue = batch[:, 2]
    print(f'2) 蓝色通道 batch[:, 2]   shape={blue.shape}   (B,H,W)')
    print(f'   数值 = {fmt(blue)}')
    print('   说明：第 1 个下标选图像、第 2 个下标选通道，剩下的 (H,W) 保留，'
          '所以 (2,3,2,2) -> (2,2,2)。')

    # 3. 转成 (B,H,W,C)
    nhwc = batch.transpose(0, 2, 3, 1)
    print(f'3) (B,H,W,C) 变换后 shape = {nhwc.shape}')
    print('   说明：transpose 只改轴的顺序、不搬数据。'
          '原来的轴编号是 (B,C,H,W)=(0,1,2,3)，重排成 (0,2,3,1) 即 (B,H,W,C)。')

    # 边界输入：单张图像也走同一套操作
    single = np.stack((a,), axis=0)
    print(f'   边界输入 只有 1 张: shape={single.shape}, '
          f'转 NHWC 后={single.transpose(0, 2, 3, 1).shape}')


# ---------------------------------------------------------------------------
# NP-M1 批量计算夹角 [中等]
# ---------------------------------------------------------------------------
def angles_to(query, X):
    """返回 query 与 X 每一行的夹角，单位度，输出 shape (N,)。

    输入：query 为 (D,)，X 为 (N,D)
    输出：np.ndarray，shape (N,)，无效行（零向量）为 np.nan
    异常：query 是零向量时抛 ValueError
    """
    query = np.asarray(query, dtype=float)
    X = np.asarray(X, dtype=float)

    norm_query = np.linalg.norm(query)
    if norm_query == 0.0:
        raise ValueError('query 是零向量，夹角无定义')

    # 一次矩阵乘得到全部点积：(N,D) @ (D,) -> (N,)
    dots = X @ query

    # 每一行的长度：(N,D) 沿最后一轴归约 -> (N,)
    row_norms = np.linalg.norm(X, axis=1)

    # 零向量行不能拿来除，用 where 只对有效行做除法，其余位置先填 nan
    cosine = np.divide(dots,
                       row_norms * norm_query,
                       out=np.full_like(dots, np.nan),
                       where=row_norms > 0.0)

    cosine = np.clip(cosine, -1.0, 1.0)   # nan 经过 clip 仍然是 nan
    return np.degrees(np.arccos(cosine))


def np_m1():
    title('NP-M1 批量计算夹角')

    query = np.array([0.60, -0.80, 0.00])
    X = np.array([[0.30, -0.40, 0.00],
                  [-0.60, 0.80, 0.00],
                  [0.80, 0.60, 0.00],
                  [0.00, 0.00, 0.00]])

    angles = angles_to(query, X)
    print(f'1) angles_to 输出 shape={angles.shape}')
    print(f'   角度 = {fmt(angles)}')
    for i, ang in enumerate(angles):
        print(f'   X[{i}] 与 query 的夹角 = {ang:.6f} 度')

    print('2) 说明：X (4,3) @ query (3,) 一步得到 4 个点积，'
          'np.linalg.norm(X, axis=1) 一步得到 4 个行长度，全程没有遍历行的循环。')

    print('3) 零向量行返回 nan（第 4 行），顺序与 X 的行顺序一致。')

    print('4) 边界输入（零查询）:')
    try:
        angles_to(np.zeros(3), X)
    except ValueError as exc:
        print(f'   抛出 ValueError: {exc}')


# ---------------------------------------------------------------------------
# NP-M2 用余弦相似度做 Top-k 检索 [中等]
# ---------------------------------------------------------------------------
def top_k_cosine(query, gallery, k):
    """按余弦相似度从高到低返回前 k 行的索引和相似度。

    输入：query (D,)，gallery (N,D)，k 为正整数
    输出：(idx, scores)，idx 为 (k,) 的 int 索引，scores 为 (k,) 的 float
    异常：query 零向量 / k 非正整数 / k 超过有效行数时抛 ValueError
    """
    query = np.asarray(query, dtype=float)
    gallery = np.asarray(gallery, dtype=float)

    norm_query = np.linalg.norm(query)
    if norm_query == 0.0:
        raise ValueError('query 是零向量，相似度无定义')

    row_norms = np.linalg.norm(gallery, axis=1)
    valid = row_norms > 0.0          # 零向量行不参与排序
    n_valid = int(valid.sum())

    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k <= 0:
        raise ValueError('k 必须是正整数')
    if k > n_valid:
        raise ValueError(f'k={k} 超过有效行数 {n_valid}')

    # 无效行填 -inf，排序后自然落到最后，不会被选中
    scores = np.full(gallery.shape[0], -np.inf)
    scores[valid] = (gallery[valid] @ query) / (row_norms[valid] * norm_query)

    # 分数相同要保留原行顺序，所以用 stable 排序
    order = np.argsort(-scores, kind='stable')
    idx = order[:k]
    return idx, scores[idx]


def np_m2():
    title('NP-M2 用余弦相似度做 Top-k 检索')

    query = np.array([0.60, -0.80, 0.00])
    gallery = np.array([[0.60, -0.80, 0.00],
                        [0.30, -0.40, 0.00],
                        [0.80, 0.60, 0.00],
                        [-0.60, 0.80, 0.00],
                        [0.00, 0.00, 0.00]])
    ids = np.array(['img_a', 'img_b', 'img_c', 'img_d', 'img_zero'])

    k = 3
    idx, scores = top_k_cosine(query, gallery, k)

    # 逐行打印完整相似度，方便核对
    print('1) query 与 5 行的余弦相似度:')
    all_scores = np.full(gallery.shape[0], np.nan)
    norms = np.linalg.norm(gallery, axis=1)
    valid = norms > 0
    all_scores[valid] = (gallery[valid] @ query) / (norms[valid] * np.linalg.norm(query))
    for i in range(gallery.shape[0]):
        shown = 'nan（零向量，不参与排序）' if np.isnan(all_scores[i]) else f'{all_scores[i]:.6f}'
        print(f'   {ids[i]:<9} score = {shown}')

    print(f'2) 点乘除以两边长度计算相似度；零向量行被排除，只剩 4 个有效行。')

    print(f'3) Top-{k} 结果（稳定排序，同分保持原顺序）:')
    for rank, (i, s) in enumerate(zip(idx, scores), start=1):
        print(f'   {rank}. {ids[i]:<9} 行索引={i}  score={s:.6f}')

    print('4) 边界输入:')
    for bad_k in (0, 4, 5):
        try:
            top_k_cosine(query, gallery, bad_k)
        except ValueError as exc:
            print(f'   k={bad_k} 抛出 ValueError: {exc}')
    try:
        top_k_cosine(np.zeros(3), gallery, 2)
    except ValueError as exc:
        print(f'   零查询 抛出 ValueError: {exc}')


# ---------------------------------------------------------------------------
# NP-M3 所有点对之间的距离矩阵 [中等]
# ---------------------------------------------------------------------------
def np_m3():
    title('NP-M3 所有点对之间的距离矩阵')

    A = np.array([[0.20, -0.40],
                  [0.80, 0.30],
                  [-0.50, 0.70]])
    B = np.array([[0.10, 0.20],
                  [-0.60, -0.20]])

    # 1. 广播构造差数组：A 新增一个轴 (3,1,2)，B 新增一个轴 (1,2,2)
    diff = A[:, None, :] - B[None, :, :]
    print(f'1) 差数组 shape = {diff.shape}   (len(A), len(B), 特征数) = (3,2,2)')
    print(f'   说明：A[:, None, :] 把 (3,2) 变成 (3,1,2)，'
          f'B[None, :, :] 把 (2,2) 变成 (1,2,2)，')
    print('   两者广播后得到 (3,2,2)：第 1 个轴挑 A 的点，第 2 个轴挑 B 的点，'
          '最后一轴是坐标差。')

    # 2. 距离矩阵：对最后一轴（坐标）求平方和再开方
    D = np.sqrt(np.sum(diff ** 2, axis=-1))
    print(f'2) D = {fmt(D)}   shape={D.shape}')

    # 3. 每行最近的 B 索引
    nearest = np.argmin(D, axis=1)
    print(f'3) 输出 shape = {D.shape}')
    for i, j in enumerate(nearest):
        print(f'   A[{i}] 最近的是 B[{j}]，距离 {D[i, j]:.6f}')
    print('   说明：axis=-1 归约的是坐标轴，得到的是 (3,2) 距离矩阵；'
          'argmin(axis=1) 在每一行内部找最小值的位置。')

    # 边界输入：A 和 B 相同，距离矩阵对角线应为 0
    Dself = np.sqrt(np.sum((A[:, None, :] - A[None, :, :]) ** 2, axis=-1))
    print(f'   边界输入 A 对自身: 对角线 = {fmt(np.diag(Dself))}（应全为 0）')


# ---------------------------------------------------------------------------
# NP-M4 用矩阵运算完成一次前向计算 [中等]
# ---------------------------------------------------------------------------
def np_m4():
    title('NP-M4 用矩阵运算完成一次前向计算')

    X = np.array([[0.80, -0.60, 0.70],
                  [-0.40, -0.50, 0.90]])
    W1 = np.array([[-0.10, 0.20, -0.80],
                   [0.30, -0.50, 0.10]])
    b1 = np.array([-0.15, 0.05])
    W2 = np.array([[0.20, -0.60],
                   [-0.80, 0.52]])
    b2 = np.array([0.10, -0.20])

    print(f'X  shape={X.shape}   W1 shape={W1.shape}   b1 shape={b1.shape}')
    print(f'W2 shape={W2.shape}  b2 shape={b2.shape}')

    # 1. 第一层线性变换。W1 按 (输出特征数, 输入特征数) 存储，所以要转置
    Z = X @ W1.T + b1
    print(f'1) Z = X @ W1.T + b1 = {fmt(Z)}   shape={Z.shape}')
    print('   说明：(2,3) @ (3,2) -> (2,2)。W1.T 是 (3,2)，'
          '正好把 3 个输入特征映射到 2 个输出特征；b1 的 (2,) 沿行广播。')

    # 2. 逐元素做 ReLU
    H = np.maximum(Z, 0.0)
    print(f'2) H = max(Z, 0) = {fmt(H)}   shape={H.shape}')
    print('   说明：np.maximum 逐元素比较，负数位置被换成 0，'
          '正数位置原样保留，形状不变。')

    # 3. 第二层线性变换
    Y = H @ W2.T + b2
    print(f'3) Y = H @ W2.T + b2 = {fmt(Y)}   shape={Y.shape}')
    print('   说明：(2,2) @ (2,2) -> (2,2)，b2 的 (2,) 沿行广播。')

    print('4) 全程只用矩阵乘、加法和逐元素比较，没有遍历样本的循环，'
          '给定参数也没有被修改。')

    # 边界输入：只有一行时结果应等于整体结果的第一行
    Y1 = X[:1] @ W1.T + b1
    Y1 = np.maximum(Y1, 0.0) @ W2.T + b2
    print(f'   边界输入 只算第一行: {fmt(Y1)}（与上面第一行一致）')


# ---------------------------------------------------------------------------
def main():
    np_e1()
    np_e2()
    np_e3()
    np_e4()
    np_e5()
    np_e6()
    np_e7()
    np_e8()
    np_m1()
    np_m2()
    np_m3()
    np_m4()
    print('\n全部 12 题运行完毕。')


if __name__ == '__main__':
    main()
