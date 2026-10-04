import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import random
import math
import os
from shapely.geometry import Polygon, box
from shapely.affinity import translate, rotate as shapely_rotate
from shapely.ops import unary_union

def get_solar_angle(t_hr):
    # 동지(Winter Solstice) 서울(위도 37.5도) 기준 태양 궤도 계산
    lat = math.radians(37.5)
    dec = math.radians(-23.44)
    h_angle = math.radians((t_hr - 12.0) * 15.0)
    
    sin_alt = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(h_angle)
    alt = math.asin(sin_alt)
    
    sin_az = -math.sin(h_angle) * math.cos(dec) / math.cos(alt)
    cos_az = (math.sin(dec) - math.sin(lat) * math.sin(alt)) / (math.cos(lat) * math.cos(alt))
    
    az = math.atan2(sin_az, cos_az)
    az_deg = math.degrees(az)
    if az_deg < 0:
        az_deg += 360.0
        
    return az_deg, math.degrees(alt)

# ==========================================================
# 동 평면 치수 산정 기준
# - 동 외곽선(발코니 포함) 바닥면적 = 전용면적 × FOOTPRINT_FACTOR × 층당 세대수
#   (공급면적[전용+주거공용] + 발코니 서비스면적 ≈ 전용 × 1.75, 판상·탑상 공통)
# - 판상형은 평형별 동 깊이를 주고 폭을 역산, 탑상형은 L자 비율(32:18)을 유지
# ==========================================================
FOOTPRINT_FACTOR = 1.75
TOWER_ASPECT = 18.0 / 32.0

def plate_dims(excl, units, depth):
    total = FOOTPRINT_FACTOR * excl * units
    return (round(total / depth, 1), float(depth))

def tower_dims(excl, units=3):
    # L자(날개 두께 = 가로/3) 면적 = b_w^2 * k
    total = FOOTPRINT_FACTOR * excl * units
    k = 1.0 / 3.0 + (TOWER_ASPECT - 1.0 / 3.0) / 3.0
    b_w = math.sqrt(total / k)
    return (round(b_w, 1), round(b_w * TOWER_ASPECT, 1))

UNIT_TYPES = {
    "26m²(복도식)": (*plate_dims(26, 4, 10.0), 4, 26.0, "판상형"),
    "31m²(복도식)": (*plate_dims(31, 4, 10.0), 4, 31.0, "판상형"),
    "36m²(복도식)": (*plate_dims(36, 4, 10.0), 4, 36.0, "판상형"),
    "41m²(복도식)": (*plate_dims(41, 4, 10.0), 4, 41.0, "판상형"),
    "46m²(복도식)": (*plate_dims(46, 4, 10.0), 4, 46.0, "판상형"),
    "55m²(계단식)": (*plate_dims(55, 4, 10.5), 4, 55.0, "판상형"),
    "59m²(계단식)": (*plate_dims(59, 4, 11.0), 4, 59.0, "판상형"),
    "65m²(계단식)": (*plate_dims(65, 4, 11.0), 4, 65.0, "판상형"),
    "74m²(계단식)": (*plate_dims(74, 4, 11.5), 4, 74.0, "판상형"),
    "84m²(계단식)": (*plate_dims(84, 4, 12.0), 4, 84.0, "판상형"),
    "84m²(탑상형)": (*tower_dims(84), 3, 84.0, "L자형"),
    "114m²(계단식)": (*plate_dims(114, 4, 13.0), 4, 114.0, "판상형"),
    "114m²(탑상형)": (*tower_dims(114), 3, 114.0, "L자형"),
}

import colorsys

def get_type_color(name):
    size = name.split('(')[0]
    hue_map = {
        "26m²": 0.75, "31m²": 0.80, "36m²": 0.00,
        "41m²": 0.05, "46m²": 0.10, "55m²": 0.15,
        "59m²": 0.50, "65m²": 0.55, "74m²": 0.08,
        "84m²": 0.60, "114m²": 0.90,
    }
    
    h = hue_map.get(size, (hash(size) % 100) / 100.0)
    
    if "계단식" in name or "복도식" in name or "판상" in name:
        l, s = 0.65, 0.7
    elif "탑상형" in name or "타워형" in name or "L자형" in name:
        l, s = 0.50, 0.85
    elif "탑상4호" in name:
        l, s = 0.40, 0.90
    else:
        l, s = 0.5, 0.7
        
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

def get_base_windows(b_w, b_l, bldg_shape, units):
    windows = []
    if bldg_shape == "판상형":
        step = b_w / units
        for i in range(units):
            windows.append((i * step, 0, (i + 1) * step, 0, 0, -1))
    elif bldg_shape in ["타워형", "L자형"]:
        windows.append((0, 0, b_w, 0, 0, -1))
        windows.append((0, 0, 0, b_l, -1, 0))
    return windows

def get_base_dividers(b_w, b_l, bldg_shape, units):
    dividers = []
    if bldg_shape == "판상형":
        for i in range(1, units):
            x = i * (b_w / units)
            dividers.append(((x, 0), (x, b_l)))
    elif bldg_shape in ["타워형", "L자형"]:
        # 탑상 3호: 세 세대 면적이 같도록 구획 (가로 날개 끝 1세대 / 모서리 1세대 / 세로 날개 끝 1세대)
        t = b_w / 3.0
        unit_len = create_building_poly(b_w, b_l, bldg_shape).area / 3.0 / t
        xa = b_w - unit_len          # 가로 날개 끝 세대 경계
        yc = b_l - unit_len          # 세로 날개 끝 세대 경계
        dividers.append(((xa, 0), (xa, t)))
        dividers.append(((0, yc), (t, yc)))
        if yc < t:                   # 세로 날개가 짧아 끝 세대가 모서리까지 내려오는 경우
            dividers.append(((t, yc), (t, t)))
    return dividers

def create_building_poly(b_w, b_l, bldg_shape):
    if bldg_shape in ["타워형", "L자형"]:
        t = b_w / 3.0
        return Polygon([(0,0), (b_w,0), (b_w, t), (t, t), (t, b_l), (0, b_l)])
    else:
        return box(0, 0, b_w, b_l)

def get_base_facades(b_w, b_l, bldg_shape):
    """채광창으로 보지 않는 긴 벽면 (측벽 제외). 형식은 windows와 같음 (x1, y1, x2, y2, 바깥방향 dx, dy)
    시행령 제86조 제3항 제2호: 측벽↔측벽 4m, 측벽↔창 없는 벽면 8m → 이 벽면 앞으로는 8m 확보"""
    if bldg_shape == "판상형":
        return [(0, b_l, b_w, b_l, 0, 1)]
    elif bldg_shape in ["타워형", "L자형"]:
        t = b_w / 3.0
        return [(t, t, b_w, t, 0, 1), (t, t, t, b_l, 1, 0)]
    return []

def transform_windows(windows, angle, tx, ty):
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    transformed = []
    for (x1, y1, x2, y2, dx, dy) in windows:
        nx1 = x1 * cos_a - y1 * sin_a + tx
        ny1 = x1 * sin_a + y1 * cos_a + ty
        nx2 = x2 * cos_a - y2 * sin_a + tx
        ny2 = x2 * sin_a + y2 * cos_a + ty
        ndx = dx * cos_a - dy * sin_a
        ndy = dx * sin_a + dy * cos_a
        transformed.append((nx1, ny1, nx2, ny2, ndx, ndy))
    return transformed

def transform_dividers(dividers, angle, tx, ty):
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    transformed = []
    for ((x1, y1), (x2, y2)) in dividers:
        nx1 = x1 * cos_a - y1 * sin_a + tx
        ny1 = x1 * sin_a + y1 * cos_a + ty
        nx2 = x2 * cos_a - y2 * sin_a + tx
        ny2 = x2 * sin_a + y2 * cos_a + ty
        transformed.append(((nx1, ny1), (nx2, ny2)))
    return transformed

def is_valid_orientation(windows):
    for (wx1, wy1, wx2, wy2, dx, dy) in windows:
        az = math.degrees(math.atan2(dx, dy))
        if az < 0: az += 360
        if not (120 <= az <= 240):
            return False
    return True

# ==========================================================
# 복합 동 타입 자동 생성 (탑상 4호)
# - 같은 평형의 판상형(계단식) 모듈과 탑상형(L자) 모듈을 그대로 재사용해서
#   세대당 바닥면적이 기존 타입과 동일하게 유지되도록 조립합니다.
# ==========================================================

# 1. 모든 평형에 대해 '타워형(L자형)' 뼈대가 없으면 자동 생성 (그래야 복합동을 만들 수 있음)
for key in list(UNIT_TYPES.keys()):
    if "(계단식)" in key or "(복도식)" in key:
        size_label = key.split("(")[0]
        area = float(size_label.replace("m²", ""))
        tower_key1 = f"{size_label}(탑상형)"
        tower_key2 = f"{size_label}(타워형)"
        if tower_key1 not in UNIT_TYPES and tower_key2 not in UNIT_TYPES:
            UNIT_TYPES[tower_key1] = (*tower_dims(area), 3, area, "L자형")

COMPOSITE_GEOM = {}  # name -> (poly, windows, dividers, facades)

# 경기도 주택조례 제5조제1호: 판상형은 4호연립 이하로 계획하거나 1개동 길이 60m 이하
# (150세대 미만 재건축·테라스형·4층 이하 예외, 지구단위계획·건축위원회가 달리 정하면 그에 따름)
PLATE_MAX_UNITS = 4
PLATE_MAX_LENGTH = 60.0

def plate_length_ok(units, length):
    return units <= PLATE_MAX_UNITS or length <= PLATE_MAX_LENGTH

def build_composite_types(unit_types, composite_geom):
    by_size = {}
    for name, spec in list(unit_types.items()):
        size_label = name.split("(")[0]
        shape = spec[4]
        if shape == "판상형" and "복도식" not in name:
            by_size.setdefault(size_label, {})['plate'] = spec
        elif shape == "L자형":
            by_size.setdefault(size_label, {})['tower'] = spec

    for size_label, d in by_size.items():
        if 'plate' not in d or 'tower' not in d:
            continue
        p_w, p_l, p_units, area, _ = d['plate']
        t_w, t_l, t_units, _, _ = d['tower']

        pw = p_w / p_units          # 판상 1세대 전면폭
        pd = p_l                    # 판상 동 깊이
        t = t_w / 3.0               # 탑상 날개 두께
        t_area = create_building_poly(t_w, t_l, "L자형").area
        u_t = (t_area / t_units) / t  # 탑상 1세대가 날개 방향으로 차지하는 길이

        # 1) 탑상 4호 : ㄱ자 양 날개 2세대 + 2세대 (세대당 면적은 탑상 3호와 동일)
        W = 2 * u_t + t / 2.0
        poly4 = Polygon([(0, 0), (W, 0), (W, t), (t, t), (t, W), (0, W)])
        win4 = [(0, 0, W, 0, 0, -1), (0, 0, 0, W, -1, 0)]
        # 네 세대 면적이 같도록 구획: 두 날개 끝 세대 + 모서리를 반씩 나눈 두 세대
        div4 = [((W - u_t, 0), (W - u_t, t)), ((0, W - u_t), (t, W - u_t)),
                ((t / 2.0, 0), (t / 2.0, t)), ((t / 2.0, t), (t, t))]
        n4 = f"{size_label}(탑상4호)"
        unit_types[n4] = (W, W, 4, area, "탑상4호")
        fac4 = [(t, t, W, t, 0, 1), (t, t, t, W, 1, 0)]
        composite_geom[n4] = (poly4, win4, div4, fac4)
        # ※ 판탑 혼합동은 2026-10-04 제거: 45° 회전 배치에서는 탑상 4호와 면적·세대수·방향이 같아
        #   가배치 단계에서 구분 실익이 없음 (사용자 결정)

build_composite_types(UNIT_TYPES, COMPOSITE_GEOM)

def get_type_group(name, shape, units):
    if shape == "판상형": return "판상"
    if shape == "L자형": return "탑상3"
    if shape == "탑상4호": return "탑상4"
    return "기타"

# ==========================================================
# 면적 산정 (주택공급에 관한 규칙 / 건축법 시행령 제119조 기준)
# - 공급면적 = 주거전용 + 주거공용(계단·EV·복도·벽체)  →  용적률 산정 연면적
# - 계약면적 = 공급면적 + 기타공용(지하주차장·관리동 등, 용적률 제외)
# - 서비스면적(발코니)은 바닥면적에서 제외되므로 용적률·공급면적에 들어가지 않음
# eff = (계단식 전용률, 탑상형 전용률, 복도식 전용률)
# ==========================================================
DEFAULT_EFF = (0.76, 0.74, 0.72)

def type_area_breakdown(name, eff=DEFAULT_EFF):
    """층당 (전용면적 합, 공급면적 합) 반환"""
    _, _, units, area, shape = UNIT_TYPES[name]
    e_stair, e_tower, e_corr = eff
    if shape == "판상형":
        e = e_corr if "복도식" in name else e_stair
        supply = area / e * units
    elif shape in ("L자형", "탑상4호"):
        supply = area / e_tower * units
    else:
        supply = area / e_stair * units
    return area * units, supply

ALL_GROUPS = ("판상", "탑상3", "탑상4")
# AI 자동 모드에서 비교하는 조합들
AUTO_COMBOS = {
    "판상형": ("판상",),
    "탑상형": ("탑상3", "탑상4"),
    "혼합": ALL_GROUPS,
}
GROUP_LABELS = {"판상": "판상형", "탑상3": "탑상 3호", "탑상4": "탑상 4호"}
# 조합 간 세대수 차이가 이 비율 이내면 '사실상 같음'으로 안내 (탑상동 그리디 배치의 무작위 편차 수준)
NEAR_TIE_RATIO = 0.03

def tower_units_of(name, shape, units):
    """층당 탑상형 세대수: 탑상 3호·4호 동은 전 세대가 탑상형"""
    if shape in ("L자형", "탑상4호"):
        return units
    return 0

def tower_share(blds):
    """세대수 기준 탑상형 비율 (0~1)"""
    total = sum(b[5] * b[4] for b in blds)
    if total == 0:
        return 0.0
    return sum(tower_units_of(b[7], b[1], b[5]) * b[4] for b in blds) / total

def type_label(name):
    """화면 표시용 동 타입 이름. 예: '84m²(탑상4호)' → '84m² 탑상 4호', '26m²(복도식)' → '26m² 판상형(복도식)'"""
    _, _, units, _, shape = UNIT_TYPES[name]
    label = GROUP_LABELS.get(get_type_group(name, shape, units), shape)
    if "복도식" in name:
        label += "(복도식)"
    return f"{name.split('(')[0]} {label}"

def describe_composition(blds):
    """조합 이름(허용 형태)이 아니라 실제로 배치된 동 형태로 구성 문자열 생성. 예: '판상형 13동 + 탑상 4호 1동'"""
    counts = {}
    for b in blds:
        g = get_type_group(b[7], b[1], b[5])
        counts[g] = counts.get(g, 0) + 1
    if not counts:
        return "배치 없음"
    order = list(GROUP_LABELS)
    return " + ".join(f"{GROUP_LABELS.get(g, g)} {n}동" for g, n in sorted(counts.items(), key=lambda x: order.index(x[0]) if x[0] in order else 99))

# ==========================================================
# 배치 엔진 상수
# ==========================================================
FLOOR_HEIGHT = 3.0
SIDE_FACADE_DIST = 8.0   # 측벽 ↔ 창 없는 벽면 (시행령 제86조 제3항 제2호). 측벽 ↔ 측벽은 side_dist(4m)
RATIO_TOL = 0.02         # 평형 비율: 목표보다 이만큼 넘게 배치된 평형은 잠시 배치 보류
GRID_MIN_STEP = 6.0      # 배치 후보점 탐색 간격 하한 (m). 대지가 크면 자동으로 넓어짐
N_TRIALS = 3             # 무작위 순서를 바꿔 여러 번 돌린 뒤 세대수 최대 결과 채택

# 학교 일조: 동지 08~16시를 30분 구간으로 나눠 (구간 중앙 시각으로 판정)
# 연속 2시간(09~15시) 이상 또는 총 4시간(08~16시) 이상 햇빛이 들면 통과
SCHOOL_TIMES = [8.25 + 0.5 * i for i in range(16)]
SCHOOL_TOTAL_MIN = 8                      # 30분 × 8 = 4시간
SCHOOL_CONT_MIN = 4                       # 30분 × 4 = 2시간
SCHOOL_CONT_RANGE = (2, 14)               # SCHOOL_TIMES 인덱스 2~13 = 09~15시
SCHOOL_SAMPLE_GAP = 10.0                  # 학교 경계 위 판정점 간격 (m)

def _shadow_unit(t_hr):
    """높이 1m당 그림자 이동량 (dx, dy)"""
    az_deg, alt_deg = get_solar_angle(t_hr)
    k = 1.0 / math.tan(math.radians(alt_deg))
    return (math.sin(math.radians(az_deg - 180)) * k, math.cos(math.radians(az_deg - 180)) * k)

SCHOOL_SHADOW_UNITS = [_shadow_unit(t) for t in SCHOOL_TIMES]

def school_sun_ok(sunny_row):
    """한 판정점의 30분 구간별 일조 여부 → 기준 충족 여부"""
    if sum(sunny_row) >= SCHOOL_TOTAL_MIN:
        return True
    run = 0
    for i in range(*SCHOOL_CONT_RANGE):
        run = run + 1 if sunny_row[i] else 0
        if run >= SCHOOL_CONT_MIN:
            return True
    return False

def _proj(w, dist):
    wx1, wy1, wx2, wy2, dx, dy = w
    return Polygon([(wx1, wy1), (wx2, wy2), (wx2 + dx * dist, wy2 + dy * dist), (wx1 + dx * dist, wy1 + dy * dist)])

@st.cache_data(show_spinner=False)
def auto_optimize_layout(site_w, site_l, site_shape_type, floors, h_multiplier, setback_x, setback_y, min_ns_dist, side_dist, max_far, max_bcr, selected_sizes, size_ratios, school_n, school_s, school_e, school_w, road_n, road_s, road_e, road_w, sunlight_dir, trap_bottom=0, trap_top=0, trap_height=0, l_w=0, l_l=0, l_w_inner=0, l_l_inner=0, eff=DEFAULT_EFF, layout_version=32, flip_h=False, flip_v=False, plate_only=False, tower_only=False, allowed_groups=None, min_floors=5, n_trials=N_TRIALS, min_tower_ratio=0.0):

    if site_shape_type == "직사각형":
        base_site_poly = Polygon([(0,0), (site_w,0), (site_w,site_l), (0,site_l)])
    elif site_shape_type == "L자형":
        base_site_poly = Polygon([(0,0), (l_w,0), (l_w, l_l - l_l_inner), (l_w - l_w_inner, l_l - l_l_inner), (l_w - l_w_inner, l_l), (0, l_l)])
    elif site_shape_type == "사다리꼴":
        offset = (trap_bottom - trap_top) / 2
        base_site_poly = Polygon([(0,0), (trap_bottom,0), (trap_bottom - offset, trap_height), (offset, trap_height)])
    elif site_shape_type == "ㄱ자형":
        base_site_poly = Polygon([(0,0), (l_w - l_w_inner, 0), (l_w - l_w_inner, l_l_inner), (l_w, l_l_inner), (l_w, l_l), (0, l_l)])
    else:
        base_site_poly = Polygon([(0,0), (site_w,0), (site_w,site_l), (0,site_l)])

    if flip_h:
        import shapely.affinity
        base_site_poly = shapely.affinity.scale(base_site_poly, xfact=-1, origin='center')
        minx, _, _, _ = base_site_poly.bounds
        base_site_poly = shapely.affinity.translate(base_site_poly, xoff=-minx)
    if flip_v:
        import shapely.affinity
        base_site_poly = shapely.affinity.scale(base_site_poly, yfact=-1, origin='center')
        _, miny, _, _ = base_site_poly.bounds
        base_site_poly = shapely.affinity.translate(base_site_poly, yoff=-miny)

    site_area = base_site_poly.area
    # 다각형의 안쪽으로 setback만큼 쪼그라든(Shrink) 실제 건축가능 영역 생성
    site_poly = base_site_poly.buffer(-setback_x)
    from shapely.prepared import prep
    import shapely
    site_poly_p = prep(site_poly)

    # ---- 사선/채광 이격 판정용 '인접대지경계선' 영역 (실제 대지 경계 기준) ----
    # 도로·공원 등이 있으면 공동주택은 그 '중심선'을 인접대지경계선으로 봄 (건축법 시행령 제86조 제6항)
    # → 해당 방향으로 대지를 (빈 땅 너비 / 2) 만큼 밀어낸 영역까지 허용
    _eps = 0.05
    if sunlight_dir == "정북방향":
        sun_region = unary_union([base_site_poly, translate(base_site_poly, yoff=road_n / 2.0)]).buffer(_eps)
    else:
        sun_region = unary_union([base_site_poly, translate(base_site_poly, yoff=-road_s / 2.0)]).buffer(_eps)
    win_region = unary_union([
        base_site_poly,
        translate(base_site_poly, yoff=road_n / 2.0),
        translate(base_site_poly, yoff=-road_s / 2.0),
        translate(base_site_poly, xoff=road_e / 2.0),
        translate(base_site_poly, xoff=-road_w / 2.0),
    ]).buffer(_eps)
    sun_region_p = prep(sun_region)
    win_region_p = prep(win_region)

    # 바운딩 박스 기준으로 학교 위치 설정
    minx, miny, maxx, maxy = base_site_poly.bounds

    # 학교 일조 판정점: 대지와 마주보는 학교 경계선 위 (가장 불리한 위치)
    school_xs, school_ys = [], []
    def _samples(a0, a1):
        n = max(2, int((a1 - a0) / SCHOOL_SAMPLE_GAP) + 1)
        return list(np.linspace(a0, a1, n))
    off_n = max(5, road_n)
    if school_n:
        for sx in _samples(minx, maxx): school_xs.append(sx); school_ys.append(maxy + off_n + 0.5)
    off_s = max(5, road_s)
    if school_s:
        for sx in _samples(minx, maxx): school_xs.append(sx); school_ys.append(miny - off_s - 0.5)
    off_e = max(5, road_e)
    if school_e:
        for sy in _samples(miny, maxy): school_xs.append(maxx + off_e + 0.5); school_ys.append(sy)
    off_w = max(5, road_w)
    if school_w:
        for sy in _samples(miny, maxy): school_xs.append(minx - off_w - 0.5); school_ys.append(sy)
    school_xs = np.array(school_xs); school_ys = np.array(school_ys)
    has_school = len(school_xs) > 0
    if has_school:
        sch_bbox = (school_xs.min(), school_ys.min(), school_xs.max(), school_ys.max())
        su_x = [u[0] for u in SCHOOL_SHADOW_UNITS]; su_y = [u[1] for u in SCHOOL_SHADOW_UNITS]
        su_minx, su_maxx, su_miny, su_maxy = min(0, min(su_x)), max(0, max(su_x)), min(0, min(su_y)), max(0, max(su_y))

    def shade_matrix(poly, b_h):
        """poly(높이 b_h)가 각 판정점·시각에 그림자를 드리우는지 (판정점 수 × 시각 수) bool 배열"""
        res = np.zeros((len(school_xs), len(SCHOOL_TIMES)), dtype=bool)
        px0, py0, px1, py1 = poly.bounds
        for ti, (ux, uy) in enumerate(SCHOOL_SHADOW_UNITS):
            sx, sy = ux * b_h, uy * b_h
            # 그림자 외곽 사각형 안에 있는 판정점만 정밀 판정
            near = ((school_xs >= px0 + min(0, sx)) & (school_xs <= px1 + max(0, sx)) &
                    (school_ys >= py0 + min(0, sy)) & (school_ys <= py1 + max(0, sy)))
            if not near.any():
                continue
            hull = shapely.MultiPoint(list(poly.exterior.coords) + [(cx + sx, cy + sy) for cx, cy in poly.exterior.coords]).convex_hull
            idx = np.where(near)[0]
            res[idx, ti] = shapely.contains_xy(hull, school_xs[idx], school_ys[idx])
        return res

    def school_ok(candidate, b_h, shaded):
        cx0, cy0, cx1, cy1 = candidate.bounds
        if (cx1 + su_maxx * b_h < sch_bbox[0] or cx0 + su_minx * b_h > sch_bbox[2] or
                cy1 + su_maxy * b_h < sch_bbox[1] or cy0 + su_miny * b_h > sch_bbox[3]):
            return True
        new_shade = shade_matrix(candidate, b_h)
        changed = np.where((new_shade & ~shaded).any(axis=1))[0]
        for pi in changed:
            if not school_sun_ok(~(shaded[pi] | new_shade[pi])):
                return False
        return True

    types_info = []

    selected_keys = [k for k in UNIT_TYPES.keys() if k.split("(")[0] in selected_sizes]
    if allowed_groups is None:
        if plate_only:
            allowed_groups = ("판상",)
        elif tower_only:
            allowed_groups = ("탑상3", "탑상4")
        else:
            allowed_groups = ALL_GROUPS
    selected_keys = [k for k in selected_keys if get_type_group(k, UNIT_TYPES[k][4], UNIT_TYPES[k][2]) in allowed_groups]
    for name in selected_keys:
        size_label = name.split("(")[0]
        if size_ratios.get(size_label, 0) <= 0:
            continue
        w, l, units, area, shape = UNIT_TYPES[name]
        if shape == "판상형" and not plate_length_ok(units, w):
            continue  # 경기도 주택조례 판상형 길이 제한 위반 타입은 사용하지 않음
        if name in COMPOSITE_GEOM:
            poly, windows, dividers, facades = COMPOSITE_GEOM[name]
        else:
            poly = create_building_poly(w, l, shape)
            windows = get_base_windows(w, l, shape, units)
            dividers = get_base_dividers(w, l, shape, units)
            facades = get_base_facades(w, l, shape)
        _excl_floor, supply_floor = type_area_breakdown(name, eff)
        types_info.append({
            'name': name, 'size_label': size_label, 'poly': poly, 'units': units, 'shape': shape,
            'area': poly.area, 'windows': windows, 'dividers': dividers, 'facades': facades,
            'supply_floor': supply_floor,  # 층당 공급면적 합 (용적률 산정용)
            'tower_units': tower_units_of(name, shape, units),  # 층당 탑상형 세대수 (탑상형 최소 비율용)
        })

    points = []
    # 대지가 커지면 연산 속도를 위해 탐색 간격을 넓힘
    step_size = max(GRID_MIN_STEP, min(maxx - minx, maxy - miny) / 40.0)
    for x in np.arange(minx + setback_x, maxx - setback_x, step_size):
        for y in np.arange(miny + setback_y, maxy - setback_y, step_size):
            points.append((x,y))

    total_ratio = sum(size_ratios.values())
    max_floor_area = max_far / 100.0 * site_area
    lo_floor = min(min_floors, floors)

    def feasible(candidate, final_windows, final_facades, b_floors, buildings, extras, shaded):
        """b_floors 층일 때 모든 법규를 만족하는지. 높이가 높을수록 불리해지므로(단조) 이진 탐색 가능"""
        b_h = b_floors * FLOOR_HEIGHT

        # 0. 학교 일조 (교육환경보호) - 기존 동 그림자와 합산해 일조시간 판정
        if has_school and not school_ok(candidate, b_h, shaded):
            return False

        # 0-1. 일조 사선제한 (건축법 시행령 제86조 제1항: 높이 10m 초과 부분은 높이의 1/2 이상)
        #      건물을 H/2 만큼 정북(정남 기준이면 정남)으로 밀어도 인접대지경계선 안에 있어야 함
        #      → 사다리꼴 빗변, L자 파인 부분 등 실제 경계 전체에 대해 판정
        sun_dy = (b_h / 2.0) if sunlight_dir == "정북방향" else -(b_h / 2.0)
        if not sun_region_p.contains(translate(candidate, yoff=sun_dy)):
            return False

        # 0-2. 채광창 방향 이격 (시행령 제86조 제3항 제1호: 창이 있는 벽면에서 '직각 방향'으로 0.5H 이상)
        window_setback = b_h * 0.5
        for w in final_windows:
            if not win_region_p.contains(_proj(w, window_setback)):
                return False

        # 1. 동 간 이격 (인동간격 · 측벽)
        c_minx, c_miny, c_maxx, c_maxy = candidate.bounds
        for (existing_bldg, _, ex_windows, _, ex_floors, _, _, _), (ex_facades, ex_side_buf) in zip(buildings, extras):
            e_minx, e_miny, e_maxx, e_maxy = existing_bldg.bounds
            ex_h = ex_floors * FLOOR_HEIGHT
            req_ns_dist = max(min_ns_dist, max(b_h, ex_h) * h_multiplier)
            reach = max(req_ns_dist, SIDE_FACADE_DIST) + 5.0

            if c_maxx < e_minx - reach or c_minx > e_maxx + reach or c_maxy < e_miny - reach or c_miny > e_maxy + reach:
                continue
            # 측벽 ↔ 측벽 최소 이격
            if ex_side_buf.intersects(candidate):
                return False
            # 채광창 ↔ 상대 동 (인동간격, 두 동 중 높은 쪽 높이 기준)
            for w in final_windows:
                if _proj(w, req_ns_dist).intersects(existing_bldg):
                    return False
            for w in ex_windows:
                if _proj(w, req_ns_dist).intersects(candidate):
                    return False
            # 창 없는 벽면 ↔ 상대 동 8m
            for w in final_facades:
                if _proj(w, SIDE_FACADE_DIST).intersects(existing_bldg):
                    return False
            for w in ex_facades:
                if _proj(w, SIDE_FACADE_DIST).intersects(candidate):
                    return False
        return True

    def run_trial(seed, init=()):
        """무작위 순서 그리디 배치. init(이미 놓인 동 목록)이 있으면 그 위에 빈자리를 채움"""
        rnd = random.Random(seed)
        pts = list(points)
        rnd.shuffle(pts)
        buildings, extras = [], []
        total_floor_area = 0.0
        bldg_area = 0.0
        placed_counts = {size: 0 for size in selected_sizes}
        shaded = np.zeros((len(school_xs), len(SCHOOL_TIMES)), dtype=bool)
        for bt, facades, t_info in init:
            buildings.append(bt)
            extras.append((facades, prep(bt[0].buffer(side_dist, resolution=2))))
            if has_school:
                shaded |= shade_matrix(bt[0], bt[4] * FLOOR_HEIGHT)
            total_floor_area += t_info['supply_floor'] * bt[4]
            bldg_area += t_info['area']
            placed_counts[t_info['size_label']] += t_info['units'] * bt[4]
        tower_u = sum(t_info['tower_units'] * bt[4] for bt, _, t_info in init)

        for (x, y) in pts:
            total_u = sum(placed_counts.values())
            # 탑상형 최소 비율 미달이면 판상형(탑상 세대 없는 동)은 이 자리에 넣지 않음 → 탑상동부터 채움
            tower_short = min_tower_ratio > 0 and (total_u == 0 or tower_u / total_u < min_tower_ratio)

            def get_priority(t):
                target_pct = size_ratios[t['size_label']] / total_ratio if total_ratio > 0 else 0
                current_pct = placed_counts[t['size_label']] / total_u if total_u > 0 else 0
                deficit = target_pct - current_pct
                density = round(t['units'] / t['area'], 5)
                return (deficit, density)

            # 동일한 우선순위(같은 평형의 다른 형태)일 때 랜덤하게 선택되도록 섞은 후 정렬
            current_types = list(types_info)
            rnd.shuffle(current_types)
            current_types = sorted(current_types, key=get_priority, reverse=True)

            for t_info in current_types:
                size = t_info['size_label']
                # 평형 비율 제어: 이미 목표 비율을 넘긴 평형은 이 자리에 넣지 않음 (작은 평형이 빈자리를 독식하는 것 방지)
                if total_u > 0 and placed_counts[size] / total_u > size_ratios[size] / total_ratio + RATIO_TOL:
                    continue
                if tower_short and t_info['tower_units'] == 0:
                    continue
                # 건폐율: 동 외곽 바닥면적(건축면적)
                if (bldg_area + t_info['area']) / site_area * 100 > max_bcr:
                    continue
                # 용적률: 지상 연면적 = 세대 공급면적(전용+주거공용) 합계 → 남은 용적으로 올릴 수 있는 최고 층
                hi_floor = min(floors, int((max_floor_area - total_floor_area) / t_info['supply_floor'] + 1e-9))
                if hi_floor < lo_floor:
                    continue

                rotations = [0, 45, 315]  # 남향, 남동향, 남서향만 검토 (속도 최적화 및 탑상형 지원)
                rnd.shuffle(rotations)

                best = None  # (층수, angle, candidate, windows, facades)
                for angle in rotations:
                    final_windows = transform_windows(t_info['windows'], angle, x, y)
                    if not is_valid_orientation(final_windows):
                        continue
                    candidate = translate(shapely_rotate(t_info['poly'], angle, origin=(0,0)), xoff=x, yoff=y)
                    if not site_poly_p.contains(candidate):
                        continue
                    final_facades = transform_windows(t_info['facades'], angle, x, y)
                    ok = lambda f: feasible(candidate, final_windows, final_facades, f, buildings, extras, shaded)
                    if not ok(lo_floor):
                        continue
                    # 가능한 최고 층수를 이진 탐색 (최고층부터 시작, 법규에 걸리면 1개 층 단위로 깎인 결과와 동일)
                    a, b = lo_floor, hi_floor
                    while a < b:
                        m = (a + b + 1) // 2
                        if ok(m):
                            a = m
                        else:
                            b = m - 1
                    if best is None or a > best[0]:
                        best = (a, angle, candidate, final_windows, final_facades)
                    if a == hi_floor:
                        break

                if best is None:
                    continue
                b_floors, angle, candidate, final_windows, final_facades = best
                b_h = b_floors * FLOOR_HEIGHT
                final_dividers = transform_dividers(t_info['dividers'], angle, x, y)
                buildings.append((candidate, t_info['shape'], final_windows, final_dividers, b_floors, t_info['units'], max(min_ns_dist, b_h * h_multiplier), t_info['name']))
                extras.append((final_facades, prep(candidate.buffer(side_dist, resolution=2))))
                if has_school:
                    shaded |= shade_matrix(candidate, b_h)
                total_floor_area += t_info['supply_floor'] * b_floors
                bldg_area += t_info['area']
                placed_counts[size] += t_info['units'] * b_floors
                tower_u += t_info['tower_units'] * b_floors
                break

        units_total = sum(placed_counts.values())
        return units_total, buildings, bldg_area

    # ==========================================================
    # 판상형 열(줄) 배치 엔진 - 무작위 없음
    # 남향 판상동을 동서 방향 '열'로 늘어놓고, 열의 위치(y)와 층수를 동적계획법(DP)으로 골라
    # 세대수를 최대화. 열 간격 = 인동간격(두 열 중 높은 쪽 기준), 열 안의 동 간격 = 측벽 이격.
    # 같은 조건이면 항상 같은 결과, 규제가 완화되면 세대수가 줄지 않음.
    # ==========================================================
    def free_x_intervals(region, y0, y1):
        """region 안에 세로 구간 [y0, y1]이 통째로 들어가는 x 구간 목록"""
        x_lo, x_hi = minx - 1.0, maxx + 1.0
        blocked = box(x_lo, y0, x_hi, y1).difference(region)
        res, cur = [], x_lo
        if not blocked.is_empty:
            parts = getattr(blocked, 'geoms', [blocked])
            spans = sorted((p.bounds[0], p.bounds[2]) for p in parts
                           if p.area > 1e-3 and p.bounds[3] - p.bounds[1] > 1e-3)
            for a, b in spans:
                if a > cur:
                    res.append((cur, a))
                cur = max(cur, b)
        if cur < x_hi:
            res.append((cur, x_hi))
        return res

    def intersect_iv(A, B):
        res, i, j = [], 0, 0
        while i < len(A) and j < len(B):
            a, b = max(A[i][0], B[j][0]), min(A[i][1], B[j][1])
            if b > a:
                res.append((a, b))
            if A[i][1] < B[j][1]:
                i += 1
            else:
                j += 1
        return res

    def run_rows(plate_types):
        """판상형 열 배치 → [(building_tuple, facades, t_info), ...]"""
        if not plate_types:
            return []
        D = max(t['poly'].bounds[3] for t in plate_types)          # 열 깊이 (가장 깊은 동 기준)
        widths = {t['name']: t['poly'].bounds[2] for t in plate_types}
        r_sum = sum(size_ratios[t['size_label']] for t in plate_types)
        W_ref = sum(widths[t['name']] * size_ratios[t['size_label']] for t in plate_types) / r_sum
        gap_side = side_dist
        Hs = list(range(lo_floor, floors + 1))
        if not Hs:
            return []
        s_minx, s_miny, s_maxx, s_maxy = site_poly.bounds
        y_step = max(1.0, (s_maxy - s_miny) / 250.0)
        ys = np.arange(s_miny, s_maxy - D + 1e-9, y_step)
        if len(ys) == 0:
            return []

        cache_c, cache_s, cache_w = {}, {}, {}
        def row_intervals(y, h):
            b_h = h * FLOOR_HEIGHT
            k = round(y, 3)
            if k not in cache_c:
                cache_c[k] = free_x_intervals(site_poly, y, y + D)                      # 대지 안 (공지 이격 포함)
            sun_dy = (b_h / 2.0) if sunlight_dir == "정북방향" else -(b_h / 2.0)
            ks = round(y + sun_dy, 3)
            if ks not in cache_s:
                cache_s[ks] = free_x_intervals(sun_region, y + sun_dy, y + sun_dy + D)  # 일조 사선 H/2
            kw = (k, h)
            if kw not in cache_w:
                cache_w[kw] = free_x_intervals(win_region, y - b_h * 0.5, y)            # 남측 채광창 0.5H
            return intersect_iv(intersect_iv(cache_c[k], cache_s[ks]), cache_w[kw])

        def n_fit(length):
            return int((length + gap_side) // (W_ref + gap_side)) if length >= W_ref else 0

        ny, nh = len(ys), len(Hs)
        val = np.zeros((ny, nh))
        for i, y in enumerate(ys):
            for j, h in enumerate(Hs):
                val[i, j] = sum(n_fit(b - a) for a, b in row_intervals(y, h)) * h

        # 학교 일조: 열 하나만 놓았을 때도 기준 미달인 (위치, 층수)는 후보에서 제외 → 학교 쪽 열은 DP가 알아서 낮게 잡음
        # (여러 열 그림자 합산은 아래 보정 단계에서 처리)
        if has_school:
            def row_school_ok(y, h):
                total = np.zeros((len(school_xs), len(SCHOOL_TIMES)), dtype=bool)
                for a, b in row_intervals(y, h):
                    n = n_fit(b - a)
                    if n == 0:
                        continue
                    x = a + ((b - a) - (n * W_ref + (n - 1) * gap_side)) / 2.0
                    for _ in range(n):
                        total |= shade_matrix(box(x, y, x + W_ref, y + D), h * FLOOR_HEIGHT)
                        x += W_ref + gap_side
                return all(school_sun_ok(~total[pi]) for pi in range(len(school_xs)))
            for i, y in enumerate(ys):
                cand = [j for j in range(nh) if val[i, j] > 0]
                if not cand:
                    continue
                lo_j, hi_j = cand[0], cand[-1]
                if row_school_ok(y, Hs[hi_j]):
                    continue
                if not row_school_ok(y, Hs[lo_j]):
                    val[i, :] = 0
                    continue
                while lo_j < hi_j:      # 통과하는 최고 층수 이진 탐색
                    m = (lo_j + hi_j + 1) // 2
                    if row_school_ok(y, Hs[m]):
                        lo_j = m
                    else:
                        hi_j = m - 1
                val[i, lo_j + 1:] = 0

        # 열 간격: 앞 열 북측면 ~ 뒷 열 남측면(채광창) ≥ 인동간격(두 열 중 높은 쪽 기준)
        gap = [[max(min_ns_dist, h_multiplier * max(hp, h) * FLOOR_HEIGHT) for h in Hs] for hp in Hs]
        best = np.full((ny, nh), -1.0)
        back = {}
        pm = np.full((ny, nh), -1.0)            # y 인덱스 ≤ i 중 best 최대값 (높이별)
        pm_i = np.full((ny, nh), -1, dtype=int)
        for i in range(ny):
            for j in range(nh):
                v = val[i, j]
                if v <= 0:
                    continue
                b_val, b_back = v, None
                for jp in range(nh):
                    k = int(math.floor((ys[i] - D - gap[jp][j] - ys[0]) / y_step + 1e-9))
                    if k >= 0 and pm[k, jp] > 0 and pm[k, jp] + v > b_val:
                        b_val, b_back = pm[k, jp] + v, (pm_i[k, jp], jp)
                best[i, j] = b_val
                if b_back is not None:
                    back[(i, j)] = b_back
            for j in range(nh):
                if i > 0 and pm[i - 1, j] >= best[i, j]:
                    pm[i, j], pm_i[i, j] = pm[i - 1, j], pm_i[i - 1, j]
                else:
                    pm[i, j], pm_i[i, j] = best[i, j], i
        if best.max() <= 0:
            return []
        i, j = np.unravel_index(int(np.argmax(best)), best.shape)
        rows = []
        node = (int(i), int(j))
        while node is not None:
            rows.append((float(ys[node[0]]), Hs[node[1]]))
            node = back.get(node)
        rows.sort()

        # 열 채우기: 평형 비율이 부족한 평형부터, 구간 가운데 정렬
        recs = []   # [t_info, x, y, floors]
        counts = {s: 0 for s in selected_sizes}
        for (y, h) in rows:
            for a, b in row_intervals(y, h):
                seq, used = [], 0.0
                while True:
                    tot = sum(counts.values())
                    order = sorted(plate_types, key=lambda t: counts[t['size_label']] / tot - size_ratios[t['size_label']] / total_ratio if tot else 0)
                    pick = None
                    for t in order:
                        if tot > 0 and counts[t['size_label']] / tot > size_ratios[t['size_label']] / total_ratio + RATIO_TOL:
                            continue
                        need = widths[t['name']] + (gap_side if seq else 0.0)
                        if used + need <= (b - a) + 1e-9:
                            pick = t
                            break
                    if pick is None:
                        break
                    used += widths[pick['name']] + (gap_side if seq else 0.0)
                    seq.append(pick)
                    counts[pick['size_label']] += pick['units'] * h
                x = a + ((b - a) - used) / 2.0
                for t in seq:
                    recs.append([t, x, y, h])
                    x += widths[t['name']] + gap_side

        def bld_poly(r):
            return translate(r[0]['poly'], xoff=r[1], yoff=r[2])

        # 학교 일조: 기준 미달이면 미달 판정점을 가장 많이 가리는 동부터 1개 층씩 낮춤
        if has_school and recs:
            sm_cache = {}
            def sm(r):
                key = (id(r), r[3])
                if key not in sm_cache:
                    sm_cache[key] = shade_matrix(bld_poly(r), r[3] * FLOOR_HEIGHT)
                return sm_cache[key]
            while recs:
                total = np.zeros((len(school_xs), len(SCHOOL_TIMES)), dtype=bool)
                for r in recs:
                    total |= sm(r)
                bad = [pi for pi in range(len(school_xs)) if not school_sun_ok(~total[pi])]
                if not bad:
                    break
                scores = [sm(r)[bad].sum() for r in recs]
                r = recs[int(np.argmax(scores))]
                r[3] -= 1
                if r[3] < lo_floor:
                    recs.remove(r)

        # 용적률 상한: 가장 높은 동(같으면 북쪽)부터 1개 층씩 낮춤
        tot_area = sum(r[0]['supply_floor'] * r[3] for r in recs)
        while recs and tot_area > max_floor_area + 1e-6:
            r = max(recs, key=lambda r: (r[3], r[2]))
            r[3] -= 1
            tot_area -= r[0]['supply_floor']
            if r[3] < lo_floor:
                tot_area -= r[0]['supply_floor'] * r[3]
                recs.remove(r)

        # 건폐율 상한: 세대수가 가장 적은 동(같으면 북쪽)부터 제외
        while recs and sum(r[0]['area'] for r in recs) / site_area * 100 > max_bcr:
            recs.remove(min(recs, key=lambda r: (r[0]['units'] * r[3], -r[2])))

        out = []
        for t, x, y, f in recs:
            b_h = f * FLOOR_HEIGHT
            bt = (bld_poly([t, x, y, f]), t['shape'], transform_windows(t['windows'], 0, x, y),
                  transform_dividers(t['dividers'], 0, x, y), f, t['units'],
                  max(min_ns_dist, b_h * h_multiplier), t['name'])
            out.append((bt, transform_windows(t['facades'], 0, x, y), t))
        return out

    plate_types = [t for t in types_info if t['shape'] == "판상형"]
    row_init = run_rows(plate_types)
    if types_info and len(plate_types) == len(types_info):
        # 판상형만: 열 배치 결과 그대로 (무작위 없음)
        buildings = [bt for bt, _, _ in row_init]
        bldg_area = sum(t['area'] for _, _, t in row_init)
        return buildings, site_area, bldg_area, base_site_poly

    # 혼합 조합: (판상 열 배치 + 빈자리 그리디 채움)과 순수 그리디 n회 중 최대
    #  → 탑상형 최소 비율이 없으면 '판상형만' 결과보다 세대수가 적게 나오지 않음
    #  → 탑상형 최소 비율이 있으면 비율을 충족한 결과를 우선 (전부 미달이면 세대수 최대)
    def rank(result):
        meets = tower_share(result[1]) >= min_tower_ratio - RATIO_TOL
        return (meets, result[0])
    best_result = run_trial(42, init=row_init) if row_init else None
    for trial in range(n_trials):
        result = run_trial(42 + trial)
        if best_result is None or rank(result) > rank(best_result):
            best_result = result
    _, buildings, bldg_area = best_result
    return buildings, site_area, bldg_area, base_site_poly

APP_TITLE = "속 터져서 내가 직접 만들어 본 공동주택 假배치"
st.set_page_config(layout="wide", page_title=APP_TITLE)

@st.dialog("📖 사용 매뉴얼", width="large")
def show_manual():
    manual_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_manual.md")
    if os.path.exists(manual_path):
        with open(manual_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.error("매뉴얼 파일(app_manual.md)을 찾을 수 없습니다.")

col_title, col_btn = st.columns([5, 1])
with col_title:
    st.title(APP_TITLE + " 😤")
with col_btn:
    st.write("")
    st.write("")
    if st.button("📖 사용 매뉴얼", width="stretch", type="secondary"):
        show_manual()
st.markdown("<div style='background-color: #ffe066; padding: 5px 15px; border-radius: 5px; display: inline-block; font-weight: 800; font-size: 1.1em; color: #333333; margin-bottom: 20px;'>© 2026 김진우 (Jinwoo Kim). All rights reserved.</div>", unsafe_allow_html=True)

col_input, col_viz = st.columns([1, 2])

with col_input:
    # ------------------------------------------------------------------
    st.header("1. 대지 조건")
    site_shape_type = st.selectbox("대지 형상", ["직사각형", "L자형", "ㄱ자형", "사다리꼴"])

    trap_bottom, trap_top, trap_height = 0, 0, 0
    l_w, l_l, l_w_inner, l_l_inner = 0, 0, 0, 0

    if site_shape_type == "직사각형":
        c_s1, c_s2 = st.columns(2)
        site_w = c_s1.number_input("가로 (m)", min_value=30, max_value=500, value=200, step=10)
        site_l = c_s2.number_input("세로 (m)", min_value=30, max_value=500, value=200, step=10)
    elif site_shape_type == "사다리꼴":
        c_s1, c_s2, c_s3 = st.columns(3)
        trap_bottom = c_s1.number_input("아랫변 (m)", min_value=30, max_value=500, value=250, step=10)
        trap_top = c_s2.number_input("윗변 (m)", min_value=10, max_value=500, value=150, step=10)
        trap_height = c_s3.number_input("높이 (m)", min_value=30, max_value=500, value=200, step=10)
        site_w = max(trap_bottom, trap_top)
        site_l = trap_height
    else:  # L자형 · ㄱ자형
        col_w1, col_w2 = st.columns(2)
        l_w = col_w1.number_input("전체 가로 (m)", min_value=30, max_value=500, value=250, step=10)
        l_w_inner = col_w2.number_input("파인 부분 가로 (m)", min_value=10, max_value=500, value=120, step=10)
        col_l1, col_l2 = st.columns(2)
        l_l = col_l1.number_input("전체 세로 (m)", min_value=30, max_value=500, value=200, step=10)
        l_l_inner = col_l2.number_input("파인 부분 세로 (m)", min_value=10, max_value=500, value=100, step=10)
        # 파인 부분이 전체보다 커지지 않게
        l_w_inner = min(l_w_inner, l_w - 10)
        l_l_inner = min(l_l_inner, l_l - 10)
        site_w = l_w
        site_l = l_l

    flip_h = False
    flip_v = False
    if site_shape_type in ["L자형", "ㄱ자형", "사다리꼴"]:
        col_f1, col_f2 = st.columns(2)
        flip_h = col_f1.checkbox("좌우 반전", value=False)
        flip_v = col_f2.checkbox("상하 반전", value=False)

    st.subheader("주변 도로·공원 너비")
    st.caption("대지 경계 밖이 도로·공원·하천이면 그 너비를 입력하세요. 0 = 다른 대지와 바로 맞닿음. "
               "공동주택은 그 중심선을 인접대지경계선으로 보므로 일조 사선·채광창 이격이 완화됩니다.")
    col_n, col_s = st.columns(2)
    road_n = col_n.number_input("북쪽 (m)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
    road_s = col_s.number_input("남쪽 (m)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
    col_e, col_w = st.columns(2)
    road_e = col_e.number_input("동쪽 (m)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
    road_w = col_w.number_input("서쪽 (m)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)

    st.subheader("인접 학교 (교육환경보호)")
    st.caption("학교가 있는 방향을 체크하세요. 학교와 대지 사이 도로·공원 너비는 위에 입력합니다. "
               "동지 기준 학교 경계에 연속 2시간 또는 총 4시간 이상 햇빛이 들도록 층수를 조정합니다.")
    col_sch_n, col_sch_s = st.columns(2)
    school_n = col_sch_n.checkbox("북쪽 학교", value=False)
    school_s = col_sch_s.checkbox("남쪽 학교", value=False)
    col_sch_e, col_sch_w = st.columns(2)
    school_e = col_sch_e.checkbox("동쪽 학교", value=False)
    school_w = col_sch_w.checkbox("서쪽 학교", value=False)

    # ------------------------------------------------------------------
    st.header("2. 개발 규모")
    col_far, col_bcr = st.columns(2)
    max_far = col_far.number_input("용적률 상한 (%)", min_value=50, max_value=1000, value=300, step=10)
    max_bcr = col_bcr.number_input("건폐율 상한 (%)", min_value=10, max_value=100, value=20, step=2,
                                   help="법정 최대(예: 60%)까지 채우면 동이 촘촘해져 사선·인동간격 때문에 오히려 층수가 깎입니다. 20\\~25% 정도가 세대수에 유리합니다.")
    limit_floors = st.checkbox("층수 제한 있음", value=True)
    col_fl1, col_fl2 = st.columns(2)
    if limit_floors:
        floors = col_fl1.number_input("최고 층수", min_value=1, max_value=100, value=35, step=1)
    else:
        floors = 50
        col_fl1.number_input("최고 층수", value=50, disabled=True, help="층수 제한이 없으면 50층까지 검토합니다.")
    min_floors = col_fl2.number_input("최저 층수", min_value=1, max_value=int(floors),
                                      value=min(int(floors), max(5, round(floors * 0.4))), step=1,
                                      help="모든 동은 최고 층수부터 시도하고 법규에 걸리면 1개 층씩 깎습니다. 이 층수 아래로 깎아야 하는 자리에는 동을 놓지 않습니다. 기본값 = 최고 층수의 40%(최소 5층).")
    with st.expander("면적 산정 기준 (전용률 · 기타공용)", expanded=False):
        st.caption("용적률은 세대별 공급면적(전용 + 주거공용) 합계로 산정합니다. 발코니(서비스면적)는 제외됩니다.")
        c_e1, c_e2, c_e3 = st.columns(3)
        eff_stair = c_e1.number_input("계단식 전용률 (%)", min_value=50, max_value=95, value=76, step=1, help="공급면적 대비 전용면적 비율. 84m² 기준 76% → 공급 약 110.5m²")
        eff_tower = c_e2.number_input("탑상형 전용률 (%)", min_value=50, max_value=95, value=74, step=1, help="코어가 커서 계단식보다 약간 낮음. 84m² 기준 74% → 공급 약 113.5m²")
        eff_corr = c_e3.number_input("복도식 전용률 (%)", min_value=50, max_value=95, value=72, step=1, help="공용복도 때문에 가장 낮음")
        other_common_pct = st.number_input("기타공용면적 (공급면적 대비 %)", min_value=0, max_value=150, value=50, step=5,
                                           help="지하주차장·관리동·커뮤니티 등. 계약면적 = 공급면적 + 기타공용. 용적률에는 들어가지 않습니다.")
    eff = (eff_stair / 100.0, eff_tower / 100.0, eff_corr / 100.0)

    # ------------------------------------------------------------------
    st.header("3. 평형 구성")
    st.caption("배치에 사용할 평형(전용면적)을 모두 고르세요.")
    st.markdown("**복도식**")
    c1, c2, c3, c4, c5 = st.columns(5)
    use_26 = c1.checkbox("26m²", value=False)
    use_31 = c2.checkbox("31m²", value=False)
    use_36 = c3.checkbox("36m²", value=False)
    use_41 = c4.checkbox("41m²", value=False)
    use_46 = c5.checkbox("46m²", value=False)
    st.markdown("**계단식**")
    c6, c7, c8, c9, c10, c11 = st.columns(6)
    use_55 = c6.checkbox("55m²", value=False)
    use_59 = c7.checkbox("59m²", value=False)
    use_65 = c8.checkbox("65m²", value=False)
    use_74 = c9.checkbox("74m²", value=False)
    use_84 = c10.checkbox("84m²", value=True)
    use_114 = c11.checkbox("114m²", value=False)

    use_custom = st.checkbox("기타 평형 직접 입력", value=False)
    if use_custom:
        c_area = st.number_input("기타 평형 전용면적 (m²)", min_value=10, max_value=300, value=135,
                                 help="판상형·탑상형(3호·4호)을 기본 평형과 같은 산정식으로 자동 생성합니다.")
        if f"{c_area}m²" in {k.split("(")[0] for k in UNIT_TYPES}:
            st.warning(f"{c_area}m²는 기본 평형에 있습니다. 위 체크박스를 사용하세요.")
            use_custom = False
        else:
            # 판상형(계단식 4호): 기본 평형과 같은 산정식, 동 깊이는 평형에 비례 (84m²≈12m, 114m²≈13m)
            p_depth = min(14.0, max(10.5, round((9.0 + c_area / 28.0) * 2) / 2))
            UNIT_TYPES[f"{c_area}m²(계단식)"] = (*plate_dims(c_area, 4, p_depth), 4, float(c_area), "판상형")
            # 탑상형(L자 3호)
            UNIT_TYPES[f"{c_area}m²(탑상형)"] = (*tower_dims(c_area), 3, float(c_area), "L자형")
            # 탑상 4호
            build_composite_types(UNIT_TYPES, COMPOSITE_GEOM)

    selected_sizes = []
    for flag, label in [(use_26, "26m²"), (use_31, "31m²"), (use_36, "36m²"), (use_41, "41m²"), (use_46, "46m²"),
                        (use_55, "55m²"), (use_59, "59m²"), (use_65, "65m²"), (use_74, "74m²"), (use_84, "84m²"),
                        (use_114, "114m²")]:
        if flag:
            selected_sizes.append(label)
    if use_custom:
        selected_sizes.append(f"{c_area}m²")

    if not selected_sizes:
        st.warning("평형을 1개 이상 선택하세요.")
        st.stop()

    st.markdown("**평형별 목표 비율 (세대수 기준, 합계 100%)**")
    ratio_cols = st.columns(len(selected_sizes))
    size_ratios = {}
    for i, size in enumerate(selected_sizes):
        default_val = 100 // len(selected_sizes)
        if i == len(selected_sizes) - 1:
            default_val += 100 % len(selected_sizes)
        size_ratios[size] = ratio_cols[i].number_input(f"{size} (%)", min_value=0, max_value=100, value=default_val, step=5)

    total_ratio = sum(size_ratios.values())
    if total_ratio != 100:
        st.error(f"비율 합계가 100%가 되어야 합니다. (현재 {total_ratio}%)")
        st.stop()

    # ------------------------------------------------------------------
    st.header("4. 동 형태")
    STYLE_OPTIONS = {
        "AI 자동": None,
        "판상형": AUTO_COMBOS["판상형"],
        "탑상형": AUTO_COMBOS["탑상형"],
        "혼합": AUTO_COMBOS["혼합"],
    }
    layout_style = st.radio("배치할 동 형태", list(STYLE_OPTIONS.keys()), horizontal=True,
                            help="· AI 자동: 판상형·탑상형·혼합 3가지 안을 모두 계산해 세대수가 가장 많은 안을 채택\n"
                                 "· 판상형: 남향 판상동(계단식 4호·복도식 4세대)을 열로 배치. 이론상 최대치 확인용\n"
                                 "· 탑상형: 탑상 3호·4호 위주. 조망 중심\n"
                                 "· 혼합: 판상형 열 배치에 탑상동(3호·4호)을 섞음. 실제 사업에 가까운 안\n\n"
                                 "※ 탑상 4호는 계단식 평형(55\\~114m²)과 기타 평형에서만 생성됩니다.")
    style_groups = STYLE_OPTIONS[layout_style]
    is_auto_style = style_groups is None
    min_tower_pct = st.number_input("탑상형 최소 비율 (세대수 기준, %)", min_value=0, max_value=100, value=0, step=5,
                                    help="기본 0% = 제한 없음 (세대수 최대 기준). 법정 비율은 없고, 지구단위계획·심의에서 비율을 요구하는 사업지만 입력하세요.\n\n"
                                         "· 입력하면 전체 세대 중 탑상동(3호·4호) 세대가 이 비율 이상이어야 하며, 혼합안은 비율을 채울 때까지 탑상동을 먼저 배치\n"
                                         "· 비율에 못 미치는 안은 비교표에 '참고용'으로만 표시하고 AI 자동에서 채택하지 않음")
    if min_tower_pct == 0:
        st.caption("※ 세대수 최대 기준입니다. 판상형 일색 배치는 건축·경관 심의에서 형태 다양화(탑상동 혼합)를 요구받을 수 있으니 비교표의 혼합·탑상형 안도 함께 참고하세요.")
    min_tower_ratio = min_tower_pct / 100.0

    # ------------------------------------------------------------------
    st.header("5. 건축 규제")
    sunlight_dir = st.radio("일조 사선제한 방향 (건축법 제61조)", ["정북방향", "정남방향"], horizontal=True, index=0,
                            help="원칙은 정북방향입니다. 택지개발지구·지구단위계획 등에서 정남방향을 적용하기도 합니다.")
    col_h, col_side = st.columns(2)
    h_multiplier = col_h.number_input("인동간격 (H 배수)", min_value=0.1, max_value=2.0, value=0.8, step=0.1,
                                      help="마주보는 두 동 중 높은 쪽 높이 기준. 광명시 조례 0.8H. 지자체 조례에 맞게 바꾸세요.")
    side_dist = col_side.number_input("측벽 간 이격 (m)", min_value=4.0, max_value=30.0, value=8.0, step=1.0,
                                      help="측벽끼리 마주보는 동 사이 거리. 법정 최소 4m이지만 고층 동을 4m만 띄우는 경우는 드물어 기본 8m. 판상형 열 안의 동 간격으로도 쓰입니다.")
    min_ns_dist = 10.0
    setback_x = 3.0
    setback_y = 3.0

    with st.expander("적용 규제 요약", expanded=False):
        st.markdown(f"""
- **일조 사선 ({sunlight_dir}):** {'북쪽' if sunlight_dir == '정북방향' else '남쪽'} 인접대지경계선에서 H/2 이상 (도로·공원은 중심선 기준)
- **채광창 이격:** 창이 있는 벽면에서 직각 방향으로 인접대지경계선까지 0.5H 이상
- **인동간격:** {h_multiplier:g}H 이상, 최소 {min_ns_dist:g} m (두 동 중 높은 쪽 기준)
- **측벽 이격:** 측벽↔측벽 {side_dist:g} m, 측벽↔창 없는 벽면 8 m 이상
- **대지 안의 공지:** 대지경계선에서 {setback_x:g} m
- **동 규모:** 4호 이하이거나 1개동 길이 60 m 이하 (경기도 주택조례 제5조제1호, 꺾인 동은 외곽 긴 변 기준)
- **학교 일조:** 동지 기준 학교 경계에 연속 2시간(09\\~15시) 또는 총 4시간(08\\~16시) 이상
""")


with col_viz:
    inputs_tuple = (
        layout_style, site_shape_type, site_w, site_l, h_multiplier, side_dist, trap_bottom, trap_top, trap_height,
        l_w, l_l, l_w_inner, l_l_inner, max_far, max_bcr,
        limit_floors, floors, min_floors, sunlight_dir, school_n, school_s, school_e, school_w,
        road_n, road_s, road_e, road_w, str(selected_sizes), str(size_ratios),
        flip_h, flip_v, str(eff), min_tower_pct
    )

    calc_btn = st.button("🚀 배치 계산", type="primary", width="stretch")

    if calc_btn:
        st.session_state['last_inputs'] = inputs_tuple
        with st.spinner("판상형·탑상형·혼합 3가지 안을 계산하고 있습니다... (대지 크기·학교 조건에 따라 수십 초\\~2분)"):
            def run_layout(groups):
                # 탑상형 최소 비율은 판상과 탑상이 섞이는 안에만 의미가 있음 (나머지는 0으로 캐시 공유)
                mixes = "판상" in groups and any(g != "판상" for g in groups)
                return auto_optimize_layout(
                    site_w, site_l, site_shape_type, floors, h_multiplier, setback_x, setback_y, min_ns_dist, side_dist, max_far, max_bcr, selected_sizes, size_ratios, school_n, school_s, school_e, school_w, road_n, road_s, road_e, road_w, sunlight_dir, trap_bottom=trap_bottom, trap_top=trap_top, trap_height=trap_height, l_w=l_w, l_l=l_l, l_w_inner=l_w_inner, l_l_inner=l_l_inner, eff=eff, flip_h=flip_h, flip_v=flip_v, allowed_groups=tuple(groups), min_floors=min_floors,
                    min_tower_ratio=min_tower_ratio if mixes else 0.0)

            def count_units(blds):
                return sum(b[5] * b[4] for b in blds)

            def is_near_tie(a, b):
                return max(a, b) > 0 and abs(a - b) / max(a, b) <= NEAR_TIE_RATIO

            # 3가지 안을 모두 계산해 비교
            combo_results = {name: run_layout(groups) for name, groups in AUTO_COMBOS.items()}
            alternatives = {k: count_units(v[0]) for k, v in combo_results.items()}
            compositions = {k: describe_composition(v[0]) for k, v in combo_results.items()}
            shares = {k: tower_share(v[0]) for k, v in combo_results.items()}
            meets = {k: alternatives[k] > 0 and shares[k] >= min_tower_ratio - RATIO_TOL for k in combo_results}
            # 탑상형 최소 비율을 충족한 안 중 세대수 최대 (충족안이 없으면 전체 중 최대)
            eligible = [k for k in alternatives if meets[k]] or list(alternatives)
            best_style = max(eligible, key=alternatives.get)
            best_units = alternatives[best_style]
            best_comp = compositions[best_style]
            ranked = sorted(alternatives.items(), key=lambda x: (-meets[x[0]], -x[1]))

            def compare_line(k, v):
                tag = ""
                if min_tower_ratio > 0:
                    tag = f" · 탑상 {shares[k]:.0%}" + ("" if meets[k] else f" ⚠️ 기준 {min_tower_pct}% 미달, 참고용")
                return f"- **{k}안 {v:,}세대** — {compositions[k]}{tag}"
            compare_txt = "**안별 비교** (실제 배치된 동 기준)\n" + "\n".join(compare_line(k, v) for k, v in ranked)
            if min_tower_ratio > 0 and not any(meets.values()):
                compare_txt = f"⚠️ 탑상형 {min_tower_pct}% 이상을 충족하는 안이 없어 세대수 기준으로 골랐습니다. 비율을 낮추거나 대지 조건을 확인하세요.\n\n" + compare_txt

            if is_auto_style:
                bldgs, s_area, b_area, base_site_poly = combo_results[best_style]
                cond = f"탑상형 {min_tower_pct}% 이상을 충족하는 안 중 " if min_tower_ratio > 0 and any(meets.values()) else ""
                msg = f"**AI 자동:** {cond}세대수가 가장 많은 **{best_style}안**을 채택했습니다.\n\n✅ **{best_comp} · {best_units:,}세대**"
                # 채택안과 다른 배치 중 최고안이 3% 이내면 안내 (같은 구성·세대수는 같은 배치라 제외)
                runner = next(((k, v) for k, v in ranked if k != best_style and meets[k] and (v, compositions[k]) != (best_units, best_comp)), None)
                if runner and is_near_tie(best_units, runner[1]):
                    msg += f"\n\n※ {runner[0]}안({runner[1]:,}세대)과 차이가 {NEAR_TIE_RATIO:.0%} 이내입니다. 선호하는 형태로 고르셔도 됩니다."
                hint = ("success", msg + "\n\n" + compare_txt)
            else:
                bldgs, s_area, b_area, base_site_poly = run_layout(style_groups)
                current_units = count_units(bldgs)
                current_comp = describe_composition(bldgs)
                cur_share = tower_share(bldgs)
                head = f"**선택한 {layout_style}안:** {current_comp} · {current_units:,}세대"
                if min_tower_ratio > 0 and current_units > 0 and cur_share < min_tower_ratio - RATIO_TOL:
                    if any(meets.values()):
                        tail = f"⚠️ 탑상형 {cur_share:.0%}로 기준 {min_tower_pct}%에 미달합니다 (참고용). 기준을 충족하는 최대안은 **{best_style}안 {best_units:,}세대**입니다."
                    else:
                        tail = f"⚠️ 탑상형 {cur_share:.0%}로 기준 {min_tower_pct}%에 미달하고, 기준을 충족하는 안이 없습니다."
                    hint = ("warning", f"{head}\n\n{tail}\n\n{compare_txt}")
                elif best_units > current_units and not is_near_tie(best_units, current_units):
                    hint = ("warning", f"{head}\n\n💡 **{best_style}안**({best_comp})으로 바꾸면 **{best_units:,}세대 (+{best_units - current_units:,})**까지 가능합니다. 'AI 자동'을 고르면 자동 적용됩니다.\n\n{compare_txt}")
                elif best_units > current_units:
                    hint = ("info", f"{head}\n\n최대안({best_style}안 {best_units:,}세대)과 차이가 {NEAR_TIE_RATIO:.0%} 이내라 사실상 최대입니다.\n\n{compare_txt}")
                else:
                    scope = "탑상형 기준을 충족하는 안 중" if min_tower_ratio > 0 and any(meets.values()) else "비교한 안 중"
                    hint = ("success", f"{head}\n\n✅ {scope} 세대수가 가장 많습니다.\n\n{compare_txt}")

            st.session_state['sim_result'] = (bldgs, s_area, b_area, base_site_poly)
            st.session_state['hint'] = hint

    if 'last_inputs' in st.session_state and st.session_state['last_inputs'] != inputs_tuple:
        st.warning("입력값이 바뀌었습니다. '🚀 배치 계산'을 다시 눌러주세요.")
        st.stop()

    if 'sim_result' not in st.session_state:
        st.info("왼쪽에서 조건을 입력한 뒤 '🚀 배치 계산'을 누르세요. 처음 계산은 수십 초 걸릴 수 있습니다.")
        st.stop()

    buildings, site_area, bldg_area, base_site_poly = st.session_state['sim_result']

    result_box = st.container()
    plot_box = st.container()
    shadow_box = st.container()
    override_box = st.container()
    area_box = st.container()

    with shadow_box:
        col_s1, col_s2 = st.columns([1, 3])
        show_shadow = col_s1.checkbox("그림자 표시", value=True)
        time_of_day = col_s2.slider("그림자 시각 (동지)", min_value=8.0, max_value=16.0, value=12.0, step=0.5, format="%.1f시")

    with override_box:
        with st.expander("동별 층수 직접 조정", expanded=False):
            st.caption("계산된 층수를 원하는 층수로 고정합니다. 조정한 층수는 사선·인동간격·학교 일조 검토를 거치지 않습니다. 0층으로 하면 그 동을 뺍니다.")
            df = pd.DataFrame([{"동": f"{i+1}동", "타입": type_label(b[7]), "계산 층수": b[4], "조정 층수": b[4], "조정": False}
                               for i, b in enumerate(buildings)])
            edited_df = st.data_editor(
                df,
                column_config={
                    "조정": st.column_config.CheckboxColumn("조정 적용", default=False),
                    "조정 층수": st.column_config.NumberColumn("조정 층수", min_value=0, max_value=100, step=1)
                },
                disabled=["동", "타입", "계산 층수"],
                hide_index=True,
                width="stretch"
            )

    final_buildings = []
    total_floor_area = 0
    total_units = 0
    actual_bldg_area = 0

    # 면적 집계 (동 타입별)
    sum_exclusive = 0.0
    sum_supply = 0.0
    area_by_type = {}

    for i, b in enumerate(buildings):
        poly, shape, windows, dividers, b_floors, units, ns_dist, name = b
        row = edited_df.iloc[i]
        # 직접 조정 (빈 칸이면 계산 층수 유지)
        if row["조정"] and not pd.isna(row["조정 층수"]):
            b_floors = int(row["조정 층수"])
        if b_floors <= 0:
            continue

        final_buildings.append((poly, shape, windows, dividers, b_floors, units, ns_dist, name))
        excl_floor, sup_floor = type_area_breakdown(name, eff=eff)
        total_floor_area += sup_floor * b_floors
        total_units += units * b_floors
        actual_bldg_area += poly.area
        sum_exclusive += excl_floor * b_floors
        sum_supply += sup_floor * b_floors

        agg = area_by_type.setdefault(name, {"동": 0, "세대": 0, "전용": 0.0, "공급": 0.0})
        agg["동"] += 1
        agg["세대"] += units * b_floors
        agg["전용"] += excl_floor * b_floors
        agg["공급"] += sup_floor * b_floors

    bcr = (actual_bldg_area / site_area) * 100 if site_area > 0 else 0
    far = (total_floor_area / site_area) * 100 if site_area > 0 else 0

    with result_box:
        st.header("배치 결과")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("총 세대수", f"{total_units:,} 세대")
        m2.metric("동 수", f"{len(final_buildings)} 동")
        m3.metric("건폐율", f"{bcr:.1f} %", help=f"상한 {max_bcr}%")
        m4.metric("용적률", f"{far:.1f} %", help=f"상한 {max_far}%")

        hint = st.session_state.get('hint')
        if hint:
            kind, msg = hint
            {"success": st.success, "warning": st.warning, "info": st.info}[kind](msg)

        notes = []
        if len(size_ratios) > 1 and total_units > 0:
            units_by_size = {}
            for t_name, a in area_by_type.items():
                sl = t_name.split("(")[0]
                units_by_size[sl] = units_by_size.get(sl, 0) + a["세대"]
            notes.append("**평형 비율:** " + " · ".join(f"{sl} 목표 {size_ratios[sl]}% → 실제 {units_by_size.get(sl, 0) / total_units:.0%}" for sl in size_ratios))
        if min_tower_pct > 0 and total_units > 0:
            _ts = tower_share(final_buildings)
            _ok = "✅" if _ts >= min_tower_ratio - RATIO_TOL else "⚠️ 미달"
            notes.append(f"**탑상형 비율:** 기준 {min_tower_pct}% 이상 → 실제 {_ts:.0%} {_ok}")
        if notes:
            st.markdown("  \n".join(notes))

    with plot_box:
        azimuth_deg, altitude_deg = get_solar_angle(time_of_day)
        fig = go.Figure()
        site_poly = base_site_poly.buffer(-setback_x)

        bx, by = base_site_poly.exterior.xy
        fig.add_trace(go.Scatter(x=list(bx), y=list(by), fill='toself', fillcolor='rgba(240, 248, 255, 1)', line=dict(color='black', width=2), hoverinfo='skip', showlegend=False))
        ix, iy = site_poly.exterior.xy
        fig.add_trace(go.Scatter(x=list(ix), y=list(iy), mode='lines', line=dict(color='red', width=1, dash='dash'), hoverinfo='skip', showlegend=False))

        minx, miny, maxx, maxy = base_site_poly.bounds

        def add_school(x0, y0, x1, y1, text, rotation):
            fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color="#e6b800", width=2), fillcolor="rgba(255, 250, 205, 0.5)")
            fig.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2, text=text, showarrow=False, font=dict(color='#8b6508', size=12), textangle=rotation)

        off_n = max(5.0, road_n)
        if school_n: add_school(minx, maxy + off_n, maxx, maxy + off_n + 100, "북쪽 학교", 0)
        off_s = max(5.0, road_s)
        if school_s: add_school(minx, miny - off_s - 100, maxx, miny - off_s, "남쪽 학교", 0)
        off_e = max(5.0, road_e)
        if school_e: add_school(maxx + off_e, miny, maxx + off_e + 100, maxy, "동쪽 학교", -90)
        off_w = max(5.0, road_w)
        if school_w: add_school(minx - off_w - 100, miny, minx - off_w, maxy, "서쪽 학교", -90)

        if show_shadow:
            shadow_x, shadow_y = [], []
            for b in final_buildings:
                poly, _, _, _, b_floors, _, _, _ = b
                b_h = b_floors * FLOOR_HEIGHT
                shadow_length = b_h / math.tan(math.radians(altitude_deg))
                shadow_dx = math.sin(math.radians(azimuth_deg - 180)) * shadow_length
                shadow_dy = math.cos(math.radians(azimuth_deg - 180)) * shadow_length
                shadow_poly = unary_union([poly, translate(poly, xoff=shadow_dx, yoff=shadow_dy)]).convex_hull
                sx, sy = shadow_poly.exterior.xy
                shadow_x.extend(list(sx) + [None])
                shadow_y.extend(list(sy) + [None])
            fig.add_trace(go.Scatter(x=shadow_x, y=shadow_y, fill='toself', fillcolor='rgba(0,0,0,0.3)', mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False))

        for name in sorted(set(b[7] for b in final_buildings), key=lambda n: (float(n.split("m²")[0]), type_label(n))):
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color=get_type_color(name), size=15, symbol='square'), name=type_label(name), showlegend=True))

        for idx, b in enumerate(final_buildings):
            poly, bldg_shape, windows, dividers, b_floors, b_units, b_ns_dist, name = b
            x, y = poly.exterior.xy
            hover_text = f"{idx+1}동 · {type_label(name)}<br>{b_floors}층 · 층당 {b_units}세대 (동 전체 {b_units * b_floors}세대)<br>인동간격 {b_ns_dist:.1f}m"
            fig.add_trace(go.Scatter(x=list(x), y=list(y), fill='toself', fillcolor=get_type_color(name), mode='lines', line=dict(color='#333333', width=1), text=hover_text, hoverinfo='text', showlegend=False))
            # 세대 구분선은 그리지 않음 (동 외곽만 표시, 세대 구성은 hover로 확인)

            for (wx1, wy1, wx2, wy2, dx, dy) in windows:
                proj_poly = Polygon([(wx1, wy1), (wx2, wy2), (wx2 + dx * b_ns_dist, wy2 + dy * b_ns_dist), (wx1 + dx * b_ns_dist, wy1 + dy * b_ns_dist)])
                px, py = proj_poly.exterior.xy
                fig.add_trace(go.Scatter(x=list(px), y=list(py), fill='toself', fillcolor='rgba(255,0,0,0.1)', mode='lines', line=dict(color='rgba(255,0,0,0.7)', width=1, dash='dot'), hoverinfo='skip', showlegend=False))

        b_minx, b_miny, b_maxx, b_maxy = base_site_poly.bounds
        fig.update_layout(
            xaxis=dict(scaleanchor="y", scaleratio=1, visible=True, range=[b_minx - 15, b_maxx + 15], title="동서 (m)"),
            yaxis=dict(visible=True, range=[b_miny - 15, b_maxy + 15], title="남북 (m, 위쪽이 북)"),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, title="동 타입", bgcolor='rgba(255,255,255,0.8)', bordercolor='black', borderwidth=1),
            hovermode='closest',
            height=850
        )
        st.plotly_chart(fig, width="stretch")
        st.caption("검은 실선 = 대지경계 · 빨간 파선 = 대지 안의 공지(3m) · 붉은 점선 영역 = 각 동 채광창 앞 인동간격 범위 · 회색 = 그림자 · 노란 영역 = 학교. "
                   "인동간격 범위가 대지 밖(도로 등)으로 나가는 것은 문제없습니다. 동 위에 마우스를 올리면 상세 정보가 나옵니다.")

    with area_box:
        st.subheader("면적 산정")
        oc = other_common_pct / 100.0
        area_rows = []
        for t_name in sorted(area_by_type, key=lambda n: (float(n.split("m²")[0]), type_label(n))):
            a = area_by_type[t_name]
            area_rows.append({
                "동 타입": type_label(t_name), "동 수": a["동"], "세대수": a["세대"],
                "세대당 공급 (m²)": round(a["공급"] / a["세대"], 1) if a["세대"] else 0.0,
                "전용면적 합계 (m²)": round(a["전용"], 1),
                "공급면적 합계 (m²)": round(a["공급"], 1),
                "계약면적 합계 (m²)": round(a["공급"] * (1 + oc), 1),
            })
        area_rows.append({
            "동 타입": "합계", "동 수": len(final_buildings), "세대수": total_units,
            "세대당 공급 (m²)": round(sum_supply / total_units, 1) if total_units else 0.0,
            "전용면적 합계 (m²)": round(sum_exclusive, 1),
            "공급면적 합계 (m²)": round(sum_supply, 1),
            "계약면적 합계 (m²)": round(sum_supply * (1 + oc), 1),
        })
        st.dataframe(pd.DataFrame(area_rows), hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.1f") for c in
                                    ["세대당 공급 (m²)", "전용면적 합계 (m²)", "공급면적 합계 (m²)", "계약면적 합계 (m²)"]})
        st.caption(f"용적률 = 공급면적(전용 + 주거공용) 합계 ÷ 대지면적, 발코니 제외. 계약면적 = 공급면적 + 기타공용 {other_common_pct}% (용적률 미산입).")
