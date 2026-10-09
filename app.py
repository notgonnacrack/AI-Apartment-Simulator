import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import random
import math
import os
from shapely.geometry import Polygon, box, LineString
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

ALL_GROUPS = ("판상", "탑상4")
# 항상 계산해 비교하는 세 안 (혼합 = 판상형·탑상형 결과 이상이 보장되는 최대안)
AUTO_COMBOS = {
    "판상형": ("판상",),
    "탑상형": ("탑상4",),
    "혼합": ALL_GROUPS,
}
GROUP_LABELS = {"판상": "판상형", "탑상4": "탑상 4호"}
# 조합 간 세대수 차이가 이 비율 이내면 '사실상 같음'으로 안내 (탑상동 그리디 배치의 무작위 편차 수준)
NEAR_TIE_RATIO = 0.03

def tower_units_of(name, shape, units):
    """층당 탑상형 세대수: 탑상동은 전 세대가 탑상형"""
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

ROW_GAP_DX_STEP = 2.0      # 줄 간격 계산 시 좌우 위치를 훑는 간격 (m). 결과에 이 값의 절반만큼 여유를 더함
_ROW_GAP_CACHE = {}       # 줄 간격 계산 결과 (동 치수·규정이 같으면 재사용)

def spacing_dist(owner_h, target_h, h_multiplier, low_ratio, min_ns_dist):
    """창이 있는 동(owner_h층)의 창 앞에 마주보는 동(target_h층)까지 필요한 거리 (건축법 시행령 제86조 제3항 제2호)
    가목: 채광창 벽면에서 두 동 중 높은 쪽 높이 × 인동간격 배수 (법제처 해석 21-0403: 두 동 높이 모두 기준)
    나목: 창이 있는 동이 더 높으면(주된 개구부가 낮은 동을 향함) '가목에도 불구하고' 10m 이상 + 낮은 동 높이 × low_ratio
    배치 엔진과 배치도 음영이 같은 규칙을 쓰도록 한 곳에 둠"""
    if low_ratio > 0 and owner_h > target_h:
        return max(min_ns_dist, 10.0, low_ratio * target_h * FLOOR_HEIGHT)
    return max(min_ns_dist, h_multiplier * max(owner_h, target_h) * FLOOR_HEIGHT)

def _proj(w, dist):
    wx1, wy1, wx2, wy2, dx, dy = w
    return Polygon([(wx1, wy1), (wx2, wy2), (wx2 + dx * dist, wy2 + dy * dist), (wx1 + dx * dist, wy1 + dy * dist)])

@st.cache_data(show_spinner=False)
def auto_optimize_layout(site_w, site_l, site_shape_type, floors, h_multiplier, setback_x, setback_y, min_ns_dist, side_dist, max_far, max_bcr, selected_sizes, size_ratios, school_n, school_s, school_e, school_w, road_n, road_s, road_e, road_w, sunlight_dir, trap_bottom=0, trap_top=0, trap_height=0, l_w=0, l_l=0, l_w_inner=0, l_l_inner=0, eff=DEFAULT_EFF, layout_version=40, flip_h=False, flip_v=False, plate_only=False, tower_only=False, allowed_groups=None, min_floors=5, min_tower_ratio=0.0, low_ratio=0.0):

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
    import shapely

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

    types_info = []

    selected_keys = [k for k in UNIT_TYPES.keys() if k.split("(")[0] in selected_sizes]
    if allowed_groups is None:
        if plate_only:
            allowed_groups = ("판상",)
        elif tower_only:
            allowed_groups = ("탑상4",)
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

    total_ratio = sum(size_ratios.values())
    max_floor_area = max_far / 100.0 * site_area
    lo_floor = min(min_floors, floors)

    # ==========================================================
    # 줄(열) 배치 엔진 - 무작위 없음
    # 동서 방향 '줄'을 남→북으로 쌓고, 줄마다 한 가지 동 형태(판상 4호 또는 탑상 4호)·같은 층수·같은 간격.
    # 줄 위치·층수·형태는 동적계획법(DP)으로 세대수가 최대가 되도록 고름.
    #  - 판상 줄: 정남향, 동 사이 = 측벽 간격
    #  - 탑상 줄: 45° 회전(남동·남서향), 일정한 피치, 앞뒤 탑상 줄은 반 피치씩 엇갈림(지그재그)
    #  - 줄 간격: 두 줄 형태 조합별로 인동간격·측벽·창 없는 벽면 규정을 가장 불리한 좌우 위치에서 만족하는 최소 거리
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
            if b >= a:
                res.append((a, b))
            if A[i][1] < B[j][1]:
                i += 1
            else:
                j += 1
        return res

    regions = {"site": site_poly, "sun": sun_region, "win": win_region}
    fx_cache = {}

    def origin_intervals(slabs, y, region_key):
        """slabs(동 기준 가로 띠 (y0, y1, x0, x1) 목록)가 모두 region 안에 들어가는 원점 x 구간"""
        res = None
        for (y0, y1, a, b) in slabs:
            key = (region_key, round(y + y0, 2), round(y + y1, 2))
            if key not in fx_cache:
                fx_cache[key] = free_x_intervals(regions[region_key], y + y0, y + y1)
            iv = [(c - a, d - b) for c, d in fx_cache[key] if d - b >= c - a]
            res = iv if res is None else intersect_iv(res, iv)
            if not res:
                return []
        return res or []

    def slabs_of(poly, n=4):
        """다각형을 가로 띠 n개로 잘라 띠마다 x 범위 (보수적 계단 근사)"""
        bx0, by0, bx1, by1 = poly.bounds
        hs = (by1 - by0) / n
        out = []
        for k in range(n):
            y0, y1 = by0 + k * hs, by0 + (k + 1) * hs
            part = poly.intersection(box(bx0 - 1, y0, bx1 + 1, y1))
            if not part.is_empty:
                out.append((y0, y1, part.bounds[0], part.bounds[2]))
        return out

    # ---- 줄 형태 준비 ----
    plate_types = [t for t in types_info if t['shape'] == "판상형"]
    tower_types = [t for t in types_info if t['shape'] == "탑상4호"]

    def tower_geom(t):
        r = shapely_rotate(t['poly'], 45, origin=(0, 0))
        ox, oy = -r.bounds[0], -r.bounds[1]
        return {'t': t, 'ox': ox, 'oy': oy, 'poly': translate(r, ox, oy),
                'windows': transform_windows(t['windows'], 45, ox, oy),
                'facades': transform_windows(t['facades'], 45, ox, oy),
                'w': r.bounds[2] - r.bounds[0], 'h': r.bounds[3] - r.bounds[1]}

    kinds = []
    if plate_types:
        D = max(t['poly'].bounds[3] for t in plate_types)
        Wp = max(t['poly'].bounds[2] for t in plate_types)
        r_sum = sum(size_ratios[t['size_label']] for t in plate_types)
        kinds.append({
            'kind': 'plate', 'depth': D, 'tower': False,
            'W_ref': sum(t['poly'].bounds[2] * size_ratios[t['size_label']] for t in plate_types) / r_sum,
            'u_ref': sum(t['units'] * size_ratios[t['size_label']] for t in plate_types) / r_sum,
            'rep': {'poly': box(0, 0, Wp, D), 'windows': [(0, 0, Wp, 0, 0, -1)], 'facades': [(0, D, Wp, D, 0, 1)], 'w': Wp, 'h': D},
        })
    if tower_types:
        geoms = {t['name']: tower_geom(t) for t in tower_types}
        rep = max(geoms.values(), key=lambda g: g['w'])
        r_sum = sum(size_ratios[t['size_label']] for t in tower_types)
        kinds.append({
            'kind': 'tower', 'depth': rep['h'], 'tower': True, 'geoms': geoms,
            'pitch': rep['w'] + side_dist,
            'u_ref': sum(t['units'] * size_ratios[t['size_label']] for t in tower_types) / r_sum,
            'body_slabs': slabs_of(rep['poly']), 'rep': rep, 'win_slabs': {},
        })
    if not kinds:
        return [], site_area, 0.0, base_site_poly
    nk = len(kinds)

    def window_dist(owner_h, target_h):
        return spacing_dist(owner_h, target_h, h_multiplier, low_ratio, min_ns_dist)

    def req_dist(h_s, h_n):
        """남쪽 줄(h_s층)과 북쪽 줄(h_n층) 사이 인동간격. 마주보는 창은 북쪽 줄의 남향 창"""
        return window_dist(h_n, h_s)

    # ---- 줄 형태 조합별 최소 줄 간격 (남쪽 줄 바닥선 → 북쪽 줄 바닥선) ----

    def pair_conflict(S, N, dx, dy, R):
        Np = translate(N['poly'], dx, dy)
        if Np.intersects(S['buf']):
            return True
        for w in N['windows']:                       # 북쪽 줄 동의 창 앞 인동간격
            if _proj((w[0] + dx, w[1] + dy, w[2] + dx, w[3] + dy, w[4], w[5]), R).intersects(S['poly']):
                return True
        for w in S['windows']:                       # 남쪽 줄 동의 창 앞 인동간격
            if _proj(w, R).intersects(Np):
                return True
        for w in N['facades']:                       # 창 없는 벽면 8m
            if _proj((w[0] + dx, w[1] + dy, w[2] + dx, w[3] + dy, w[4], w[5]), SIDE_FACADE_DIST).intersects(S['poly']):
                return True
        for w in S['facades']:
            if _proj(w, SIDE_FACADE_DIST).intersects(Np):
                return True
        return False

    def row_gap(ks, kn, R):
        S, N = dict(kinds[ks]['rep']), kinds[kn]['rep']
        # 같은 동 형태·치수·규정이면 계산 결과가 같으므로 모듈 전역에 캐시 (세 안·층수 역산 반복 계산 가속)
        key = (kinds[ks]['kind'], tuple(round(v, 2) for v in S['poly'].bounds),
               kinds[kn]['kind'], tuple(round(v, 2) for v in N['poly'].bounds), round(side_dist, 2), round(R, 2))
        if key in _ROW_GAP_CACHE:
            return _ROW_GAP_CACHE[key]
        if kinds[ks]['kind'] == 'plate' and kinds[kn]['kind'] == 'plate':
            g = S['h'] + R                            # 판상 → 판상: 북측면에서 남측 창까지 R
        else:
            S['buf'] = S['poly'].buffer(side_dist, resolution=2)
            reach = R + side_dist + 5.0
            g = 0.0
            # 같은 줄의 동이 좌우 어디에 있어도 성립하도록, 가로 위치(dx)를 훑어 가장 큰 필요 간격을 채택.
            # 45° 벽면 때문에 dx가 1m 바뀌면 필요 간격도 최대 1m 바뀜 → 2m 간격으로 훑고 1m 여유를 더해 놓친 위치까지 보장
            for dx in np.arange(-N['w'] - reach, S['w'] + reach + 1e-9, ROW_GAP_DX_STEP):
                lo_dy, hi_dy = 0.0, S['h'] + 2 * R + N['h'] + 10.0
                if not pair_conflict(S, N, dx, lo_dy, R):
                    continue
                while hi_dy - lo_dy > 0.25:
                    mid = (lo_dy + hi_dy) / 2.0
                    if pair_conflict(S, N, dx, mid, R):
                        lo_dy = mid
                    else:
                        hi_dy = mid
                g = max(g, hi_dy)
            if g > 0:
                g += ROW_GAP_DX_STEP / 2.0
            g = max(g, S['h'])                        # 줄이 겹치지는 않게
        _ROW_GAP_CACHE[key] = g
        return g

    # ---- 줄 하나가 (y, 층수)에서 놓을 수 있는 동 위치 ----
    def plate_intervals(y, h):
        k = kinds[0]
        b_h = h * FLOOR_HEIGHT
        sun_dy = (b_h / 2.0) if sunlight_dir == "정북방향" else -(b_h / 2.0)
        iv = origin_intervals([(0, k['depth'], 0, 0)], y, "site")                  # 대지 안 (공지 이격 포함)
        iv = intersect_iv(iv, origin_intervals([(sun_dy, sun_dy + k['depth'], 0, 0)], y, "sun"))   # 일조 사선 H/2
        iv = intersect_iv(iv, origin_intervals([(-b_h * 0.5, 0, 0, 0)], y, "win"))                # 남측 채광창 0.5H
        return iv                                    # 동이 놓일 '공간' 구간

    def tower_origins(kind, y, h):
        b_h = h * FLOOR_HEIGHT
        sun_dy = (b_h / 2.0) if sunlight_dir == "정북방향" else -(b_h / 2.0)
        if h not in kind['win_slabs']:
            rep = kind['rep']
            kind['win_slabs'][h] = slabs_of(unary_union([_proj(w, b_h * 0.5) for w in rep['windows']]))
        iv = origin_intervals(kind['body_slabs'], y, "site")
        iv = intersect_iv(iv, origin_intervals([(a + sun_dy, b + sun_dy, c, d) for a, b, c, d in kind['body_slabs']], y, "sun"))
        iv = intersect_iv(iv, origin_intervals(kind['win_slabs'][h], y, "win"))   # 남동·남서 창 0.5H
        return iv                                    # 탑상동 '원점' 구간

    # ---- 동별 층수: 줄의 동 형태·간격은 일정, 층수는 동마다 그 자리에서 가능한 최고 층수 (줄 최고 층수 이하) ----
    iv_cache = {}

    def place_iv(ki, y, h):
        """형태 ki 줄을 y에 h층으로 놓을 때 쓸 수 있는 구간 (판상: 동이 놓일 공간, 탑상: 동 원점)"""
        key = (ki, round(y, 3), h)
        if key not in iv_cache:
            iv_cache[key] = plate_intervals(y, h) if kinds[ki]['kind'] == 'plate' else tower_origins(kinds[ki], y, h)
        return iv_cache[key]

    def slot_hmax(ki, y, x0, x1, h_top):
        """[x0, x1] 자리(판상: 동이 차지하는 x 범위, 탑상: 원점 x0 = x1)에서 h_top 이하로 가능한 최고 층수 (불가하면 0).
        층수가 높을수록 일조 사선·채광 이격이 불리해지므로(단조) 이진 탐색"""
        def ok(h):
            return any(a - 1e-6 <= x0 and x1 <= b + 1e-6 for a, b in place_iv(ki, y, h))
        if h_top < lo_floor or not ok(lo_floor):
            return 0
        lo_h, hi_h = lo_floor, h_top
        while lo_h < hi_h:
            m = (lo_h + hi_h + 1) // 2
            if ok(m):
                lo_h = m
            else:
                hi_h = m - 1
        return lo_h

    slot_cache = {}

    def ref_slots(ki, y):
        """줄의 대표 동 자리 [(x0, x1, 자리 최고 층수)] - 최저 층수로 놓을 수 있는 구간에 일정 간격으로 배치"""
        key = (ki, round(y, 3))
        if key in slot_cache:
            return slot_cache[key]
        kind = kinds[ki]
        out = []
        for a, b in place_iv(ki, y, lo_floor):
            if kind['kind'] == 'plate':
                W = kind['W_ref']
                n = int((b - a + side_dist) // (W + side_dist)) if b - a >= W else 0
                x = a + ((b - a) - (n * W + (n - 1) * side_dist)) / 2.0
                for _ in range(n):
                    out.append((x, x + W, slot_hmax(ki, y, x, x + W, floors)))
                    x += W + side_dist
            else:
                P = kind['pitch']
                n = int((b - a) // P) + 1
                x = a + ((b - a) - (n - 1) * P) / 2.0
                for _ in range(n):
                    out.append((x, x, slot_hmax(ki, y, x, x, floors)))
                    x += P
        slot_cache[key] = out
        return out

    def slot_poly(ki, y, x0):
        kind = kinds[ki]
        return box(x0, y, x0 + kind['W_ref'], y + kind['depth']) if kind['kind'] == 'plate' else translate(kind['rep']['poly'], x0, y)

    Hs = list(range(lo_floor, floors + 1))
    if not Hs:
        return [], site_area, 0.0, base_site_poly
    s_minx, s_miny, s_maxx, s_maxy = site_poly.bounds
    y_step = max(1.0, (s_maxy - s_miny) / 200.0)
    ys = np.arange(s_miny, s_maxy + 1e-9, y_step)
    ny, nh = len(ys), len(Hs)
    H_arr = np.array(Hs)

    # 줄 가치: 세대수 = Σ 동별 (층당 세대 × min(그 자리 최고 층수, 줄 최고 층수))
    val = np.zeros((nk, ny, nh))
    for ki, kind in enumerate(kinds):
        for i, y in enumerate(ys):
            if y + kind['depth'] > s_maxy + 1e-6:
                continue
            for (_, _, hm) in ref_slots(ki, y):
                if hm >= lo_floor:
                    val[ki, i, :] += kind['u_ref'] * np.minimum(H_arr, hm)

    # 학교 일조: 줄 하나만 놓았을 때도 기준 미달인 (위치, 줄 최고 층수)는 제외 → 학교 쪽 줄은 DP가 알아서 낮춤
    if has_school:
        def row_school_ok(ki, y, h):
            total = np.zeros((len(school_xs), len(SCHOOL_TIMES)), dtype=bool)
            for (x0, _, hm) in ref_slots(ki, y):
                if hm >= lo_floor:
                    total |= shade_matrix(slot_poly(ki, y, x0), min(hm, h) * FLOOR_HEIGHT)
            return all(school_sun_ok(~total[pi]) for pi in range(len(school_xs)))
        for ki in range(nk):
            for i, y in enumerate(ys):
                cand = [j for j in range(nh) if val[ki, i, j] > 0]
                if not cand:
                    continue
                lo_j, hi_j = cand[0], cand[-1]
                if row_school_ok(ki, y, Hs[hi_j]):
                    continue
                if not row_school_ok(ki, y, Hs[lo_j]):
                    val[ki, i, :] = 0
                    continue
                while lo_j < hi_j:      # 통과하는 최고 층수 이진 탐색
                    m = (lo_j + hi_j + 1) // 2
                    if row_school_ok(ki, y, Hs[m]):
                        lo_j = m
                    else:
                        hi_j = m - 1
                val[ki, i, lo_j + 1:] = 0

    def run_engine(robust):
        """줄 간격 계산 방식 하나로 DP·줄 채우기·후처리까지 실행 → (세대수, 동 목록, 건축면적)
        robust=False: 줄 최고 층수 그대로 간격 계산 (밀도 높음, 동별 층수 조정 후 위반은 최종 확인 단계에서 보정)
        robust=True : 동별 층수가 줄 최고보다 낮아지는 모든 경우를 감안한 간격 (나목 배수가 클 때 유리)"""
        def gap_R(h_s, h_n):
            if not robust or low_ratio <= 0:
                return req_dist(h_s, h_n)
            # 실제 층수 s ≤ h_s, n ≤ h_n 의 모든 조합 중 가장 큰 필요 거리
            #  - n ≤ s (가목): 인동간격 배수 × s ≤ … × h_s
            #  - n > s (나목): 낮은 동 배수 × s, s ≤ min(h_s, h_n - 1)
            return max(min_ns_dist, 10.0, FLOOR_HEIGHT * max(h_multiplier * h_s, low_ratio * min(h_s, h_n - 1)))
        # 줄 간격을 y 칸 수로 (남쪽 줄 형태 ks·층 jp → 북쪽 줄 형태 kn·층 j)
        off = np.zeros((nk, nh, nk, nh), dtype=int)
        for ks in range(nk):
            for kn in range(nk):
                for jp in range(nh):
                    for j in range(nh):
                        g = row_gap(ks, kn, gap_R(Hs[jp], Hs[j]))   # jp = 남쪽 줄, j = 북쪽 줄
                        off[ks, jp, kn, j] = int(math.ceil(g / y_step - 1e-9))

        tower_flag = np.array([1.0 if k['tower'] else 0.0 for k in kinds])

        def solve(lam):
            """세대수 + lam × 탑상 세대수를 최대화하는 줄 구성"""
            score = val * (1.0 + lam * tower_flag)[:, None, None]
            best = np.full((nk, ny, nh), -1.0)
            back = {}
            pm = np.full((nk, nh, ny), -1.0)            # y 인덱스 ≤ i 중 best 최대값 (형태·층수별)
            pm_i = np.full((nk, nh, ny), -1, dtype=int)
            jr = np.arange(nh)
            for i in range(ny):
                for kn in range(nk):
                    for j in range(nh):
                        v = score[kn, i, j]
                        if v <= 0:
                            continue
                        b_val, b_back = v, None
                        for ks in range(nk):
                            idx = i - off[ks, :, kn, j]
                            ok = idx >= 0
                            if not ok.any():
                                continue
                            cand = np.where(ok, pm[ks, jr, np.maximum(idx, 0)], -1.0)
                            jp = int(np.argmax(cand))
                            if cand[jp] > 0 and cand[jp] + v > b_val:
                                b_val, b_back = cand[jp] + v, (ks, int(pm_i[ks, jp, idx[jp]]), jp)
                        best[kn, i, j] = b_val
                        if b_back is not None:
                            back[(kn, i, j)] = b_back
                for k in range(nk):
                    cur = best[k, i, :]
                    if i > 0:
                        prev = pm[k, :, i - 1]
                        keep = prev >= cur
                        pm[k, :, i] = np.where(keep, prev, cur)
                        pm_i[k, :, i] = np.where(keep, pm_i[k, :, i - 1], i)
                    else:
                        pm[k, :, i] = cur
                        pm_i[k, :, i] = i
            if best.max() <= 0:
                return []
            node = tuple(int(v) for v in np.unravel_index(int(np.argmax(best)), best.shape))
            rows = []
            while node is not None:
                kn, i, j = node
                rows.append((float(ys[i]), Hs[j], kn))
                nb = back.get(node)
                node = (nb[0], nb[1], nb[2]) if nb else None
            rows.sort()
            return rows

        def rows_share(rows):
            tot = sum(val[k, int(round((y - ys[0]) / y_step)), Hs.index(h)] for y, h, k in rows)
            tw = sum(val[k, int(round((y - ys[0]) / y_step)), Hs.index(h)] for y, h, k in rows if kinds[k]['tower'])
            return tw / tot if tot > 0 else 0.0

        # 탑상형 최소 비율: 탑상 줄에 가중치(lam)를 주어 비율을 채우는 최소 가중치를 이분 탐색
        rows = solve(0.0)
        if min_tower_ratio > 0 and nk == 2 and rows_share(rows) < min_tower_ratio - RATIO_TOL:
            lo_l, hi_l = 0.0, 1.0
            best_rows = None
            while hi_l <= 64:
                r_hi = solve(hi_l)
                if rows_share(r_hi) >= min_tower_ratio - RATIO_TOL:
                    best_rows = r_hi
                    break
                lo_l, hi_l = hi_l, hi_l * 2
            if best_rows is None:
                rows = r_hi                              # 끝까지 못 채우면 탑상이 가장 많은 구성
            else:
                for _ in range(8):
                    mid = (lo_l + hi_l) / 2.0
                    r_mid = solve(mid)
                    if rows_share(r_mid) >= min_tower_ratio - RATIO_TOL:
                        hi_l, best_rows = mid, r_mid
                    else:
                        lo_l = mid
                rows = best_rows

        # ---- 줄 간격 고르게: DP는 세대수가 같은 여러 위치 중 하나를 고르므로 남는 여유가 한 줄 사이에 몰릴 수 있음
        #      (정남방향처럼 북쪽 사선이 없으면 두드러짐) → 줄 구성(형태·층수)과 세대수는 그대로 두고,
        #      각 줄이 갈 수 있는 가장 남쪽·가장 북쪽 위치의 가운데로 옮겨 여유를 줄 사이에 고르게 나눔
        if len(rows) > 1:
            idx = [int(round((y - ys[0]) / y_step)) for y, _, _ in rows]
            js = [Hs.index(h) for _, h, _ in rows]
            ks_ = [k for _, _, k in rows]
            need = [val[ks_[n], idx[n], js[n]] - 1e-9 for n in range(len(rows))]
            feas = [np.where(val[ks_[n], :, js[n]] >= need[n])[0] for n in range(len(rows))]
            gap_i = [off[ks_[n - 1], js[n - 1], ks_[n], js[n]] for n in range(1, len(rows))]
            early, late = [], [None] * len(rows)
            for n in range(len(rows)):                       # 가장 남쪽으로 붙인 위치
                lo_i = feas[n][0] if n == 0 else early[-1] + gap_i[n - 1]
                cand = feas[n][feas[n] >= lo_i]
                early.append(int(cand[0]) if len(cand) else idx[n])
            for n in range(len(rows) - 1, -1, -1):          # 가장 북쪽으로 붙인 위치
                hi_i = feas[n][-1] if n == len(rows) - 1 else late[n + 1] - gap_i[n]
                cand = feas[n][feas[n] <= hi_i]
                late[n] = int(cand[-1]) if len(cand) else idx[n]
            new_idx, prev = [], None
            for n in range(len(rows)):
                lo_i = early[n] if prev is None else max(early[n], prev + gap_i[n - 1])
                cand = feas[n][(feas[n] >= lo_i) & (feas[n] <= max(late[n], lo_i))]
                if not len(cand):
                    new_idx = idx
                    break
                target = (early[n] + late[n]) / 2.0
                pick = int(cand[np.argmin(np.abs(cand - target))])
                new_idx.append(pick)
                prev = pick
            rows = [(float(ys[i]), h, k) for i, (_, h, k) in zip(new_idx, rows)]

        # ---- 줄 채우기: 평형 비율이 부족한 평형부터, 판상은 가운데 정렬, 탑상은 앞 탑상 줄과 반 피치 엇갈림 ----
        #      동 자리는 최저 층수로 놓을 수 있는 구간에 일정 간격으로 잡고, 동마다 그 자리 최고 층수(줄 최고 층수 이하)를 줌
        counts = {s: 0 for s in selected_sizes}
        recs = []                                        # 동 목록 {'t', 'poly', 'angle', 'X', 'Y', 'h', 'y'}
        prev_tower_x = None

        def next_type(types, fits):
            tot = sum(counts.values())
            order = sorted(types, key=lambda t: counts[t['size_label']] / tot - size_ratios[t['size_label']] / total_ratio if tot else 0)
            for t in order:
                if tot > 0 and counts[t['size_label']] / tot > size_ratios[t['size_label']] / total_ratio + RATIO_TOL:
                    continue
                if fits(t):
                    return t
            # 모든 평형이 목표를 넘었으면 빈칸으로 두지 않고 가장 부족한 평형 중 들어가는 것으로 채움 (줄 간격 일정 유지)
            return next((t for t in order if fits(t)), None)

        for (y, h, ki) in rows:
            kind = kinds[ki]
            if kind['kind'] == 'plate':
                for a, b in place_iv(ki, y, lo_floor):
                    seq, used = [], 0.0
                    while True:
                        pick = next_type(plate_types, lambda t: used + t['poly'].bounds[2] + (side_dist if seq else 0.0) <= (b - a) + 1e-9)
                        if pick is None:
                            break
                        used += pick['poly'].bounds[2] + (side_dist if seq else 0.0)
                        seq.append(pick)
                        counts[pick['size_label']] += pick['units'] * h
                    x = a + ((b - a) - used) / 2.0
                    for t in seq:
                        w = t['poly'].bounds[2]
                        hb = slot_hmax(ki, y, x, x + w, h)
                        if hb >= lo_floor:
                            recs.append({'t': t, 'poly': translate(t['poly'], x, y), 'angle': 0, 'X': x, 'Y': y, 'h': hb, 'y': y})
                        counts[t['size_label']] += t['units'] * (hb - h)
                        x += w + side_dist
            else:
                P, rep = kind['pitch'], kind['rep']
                first_x = None
                for a, b in place_iv(ki, y, lo_floor):
                    n = int((b - a) // P) + 1
                    slack = (b - a) - (n - 1) * P
                    start = a + slack / 2.0
                    if prev_tower_x is not None:          # 앞 탑상 줄과 반 피치 엇갈리게
                        cand = a + ((prev_tower_x + P / 2.0 - a) % P)
                        if cand <= a + slack + 1e-9:
                            start = cand
                    for m in range(n):
                        xs = start + m * P
                        hb = slot_hmax(ki, y, xs, xs, h)
                        if hb < lo_floor:
                            continue
                        t = next_type(tower_types, lambda t: True)
                        if t is None:
                            continue
                        g = kind['geoms'][t['name']]
                        X, Y = xs + (rep['w'] - g['w']) / 2.0, y
                        recs.append({'t': t, 'poly': translate(g['poly'], X, Y), 'angle': 45, 'X': X + g['ox'], 'Y': Y + g['oy'], 'h': hb, 'y': y})
                        counts[t['size_label']] += t['units'] * hb
                        if first_x is None:
                            first_x = xs
                if first_x is not None:
                    prev_tower_x = first_x

        # 학교 일조: 기준 미달이면 미달 판정점을 가장 많이 가리는 동부터 1개 층씩 낮춤
        if has_school and recs:
            sm_cache = {}
            def rec_shade(r):
                key = (id(r), r['h'])
                if key not in sm_cache:
                    sm_cache[key] = shade_matrix(r['poly'], r['h'] * FLOOR_HEIGHT)
                return sm_cache[key]
            while recs:
                total = np.zeros((len(school_xs), len(SCHOOL_TIMES)), dtype=bool)
                for r in recs:
                    total |= rec_shade(r)
                bad = [pi for pi in range(len(school_xs)) if not school_sun_ok(~total[pi])]
                if not bad:
                    break
                r = recs[int(np.argmax([rec_shade(r)[bad].sum() for r in recs]))]
                r['h'] -= 1
                if r['h'] < lo_floor:
                    recs.remove(r)

        # 용적률 상한: 가장 높은 동(같으면 북쪽)부터 1개 층씩 낮춤
        tot_area = sum(r['t']['supply_floor'] * r['h'] for r in recs)
        while recs and tot_area > max_floor_area + 1e-6:
            r = max(recs, key=lambda r: (r['h'], r['y']))
            r['h'] -= 1
            tot_area -= r['t']['supply_floor']
            if r['h'] < lo_floor:
                tot_area -= r['t']['supply_floor'] * r['h']
                recs.remove(r)

        # 최종 동 간격 확인: 동별 층수 조정(자리·학교·용적률) 뒤 실제 층수로 모든 동 쌍의 인동간격을 다시 확인하고,
        # 위반이면 간격을 줄이는 쪽 동을 1개 층씩 낮춤 (나목 완화는 '창이 있는 동이 더 높을 때'만 성립하므로 필수)
        #  - 나목 상황(창 있는 동이 더 높음): 낮은 동을 낮춤 → 필요 거리 감소
        #  - 가목 상황: 두 동 중 높은 동을 낮춤
        for r in recs:
            r['win'] = transform_windows(r['t']['windows'], r['angle'], r['X'], r['Y'])
        max_reach = max([window_dist(floors, floors), window_dist(floors, lo_floor)] + [0.0])
        for _ in range(2000):
            viol = None
            for A in recs:
                ax0, ay0, ax1, ay1 = A['poly'].bounds
                for B in recs:
                    if B is A:
                        continue
                    bx0, by0, bx1, by1 = B['poly'].bounds
                    if bx0 > ax1 + max_reach or bx1 < ax0 - max_reach or by0 > ay1 + max_reach or by1 < ay0 - max_reach:
                        continue
                    R = window_dist(A['h'], B['h'])
                    if any(_proj(w, R - 0.01).intersection(B['poly']).area > 0.01 for w in A['win']):
                        viol = (A, B)
                        break
                if viol:
                    break
            if viol is None:
                break
            A, B = viol

            def violates(ha, hb):
                R = window_dist(ha, hb)
                return any(_proj(w, R - 0.01).intersection(B['poly']).area > 0.01 for w in A['win'])

            if low_ratio > 0 and A['h'] > B['h']:
                # 나목 상황: (1) 낮은 동 B를 낮추거나 (2) 창 있는 동 A를 B와 같은 높이로 내려 가목으로 바꾸는 것 중 세대 손실이 작은 쪽
                hb = B['h']
                while hb >= lo_floor and violates(A['h'], hb):
                    hb -= 1
                loss1 = (B['h'] - hb) * B['t']['units'] if hb >= lo_floor else B['h'] * B['t']['units']
                loss2 = (A['h'] - B['h']) * A['t']['units'] if not violates(B['h'], B['h']) else float('inf')
                if loss2 < loss1:
                    A['h'] = B['h']
                elif hb >= lo_floor:
                    B['h'] = hb
                else:
                    recs.remove(B)
                continue
            low = A if A['h'] > B['h'] else B          # 가목 상황: 두 동 중 높은 동을 낮춤
            low['h'] -= 1
            if low['h'] < lo_floor:
                recs.remove(low)

        # 건폐율 상한: 북쪽 줄의 동부터 제외
        while recs and sum(r['t']['area'] for r in recs) / site_area * 100 > max_bcr:
            recs.remove(max(recs, key=lambda r: (r['y'], r['X'])))

        buildings = []
        for r in recs:
            t, f = r['t'], r['h']
            buildings.append((r['poly'], t['shape'], transform_windows(t['windows'], r['angle'], r['X'], r['Y']),
                              transform_dividers(t['dividers'], r['angle'], r['X'], r['Y']), f, t['units'],
                              max(min_ns_dist, f * FLOOR_HEIGHT * h_multiplier), t['name']))
        bldg_area = sum(r['t']['area'] for r in recs)
        return sum(r['t']['units'] * r['h'] for r in recs), buildings, bldg_area

    results = [run_engine(False)] + ([run_engine(True)] if low_ratio > 0 else [])
    def rank(res):
        meets = tower_share(res[1]) >= min_tower_ratio - RATIO_TOL if min_tower_ratio > 0 and nk == 2 else True
        return (meets, res[0])
    _, buildings, bldg_area = max(results, key=rank)
    return buildings, site_area, bldg_area, base_site_poly


APP_TITLE = "속 터져서 내가 직접 만들어 본 공동주택 假배치"
# 버전 규칙: 배치 엔진·구조가 크게 바뀌면 앞자리(+1.0), 기능 추가·수정은 뒷자리(+0.1). 수정할 때마다 날짜와 함께 갱신
APP_VERSION = "v2.5"
APP_UPDATED = "2026-10-09"
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
    st.title(APP_TITLE)
    st.caption(f"{APP_VERSION} · {APP_UPDATED} 수정")
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
                                 help="판상형(계단식 4호)·탑상형(4호)을 기본 평형과 같은 산정식으로 자동 생성합니다.")
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
    # 배치도에 그릴 안. 세 안은 항상 모두 계산해 비교표로 보여줌
    STYLE_OPTIONS = {"탑상형 (기본)": "탑상형", "혼합": "혼합", "판상형": "판상형"}
    layout_style = st.radio("배치도에 그릴 안", list(STYLE_OPTIONS.keys()), horizontal=True,
                            help="세 안을 모두 계산해 비교표로 보여주고, 여기서 고른 안을 배치도에 그립니다.\n\n"
                                 "모든 안은 동서 방향 '줄' 단위로 배치합니다. 한 줄은 한 가지 동 형태·같은 간격이고, 층수는 동마다 그 자리에서 가능한 최고 층수입니다.\n\n"
                                 "· 탑상형 (기본): 탑상 4호 줄만 (45° 배치, 앞뒤 줄 지그재그). 요즘 단지 구성에 가까운 안\n"
                                 "· 혼합: 판상 줄과 탑상 줄을 섞어 세대수가 가장 많은 구성 (탑상형 최소 비율 충족)\n"
                                 "· 판상형: 남향 판상동(계단식 4호·복도식 4세대) 줄만\n\n"
                                 "※ 탑상 4호는 계단식 평형(55\\~114m²)과 기타 평형에서만 생성됩니다.")
    plan_key = STYLE_OPTIONS[layout_style]
    min_tower_pct = st.number_input("탑상형 최소 비율 (세대수 기준, %)", min_value=0, max_value=100, value=50, step=5,
                                    help="기본 50% (요즘 단지 구성 기준). 법정 비율은 없으므로 세대수 최대를 보려면 0%로 바꾸세요. 지구단위계획·심의에서 비율을 정한 사업지는 그 값을 입력하세요.\n\n"
                                         "· 입력하면 전체 세대 중 탑상동 세대가 이 비율 이상이어야 하며, 혼합안은 이 비율을 채우도록 탑상 줄 수를 정함\n"
                                         "· 비율에 못 미치는 안은 비교표에 '참고용'으로 표시")
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
    low_ratio = st.number_input("남저북고 완화: 낮은 동 높이 배수 (0 = 적용 안 함)", min_value=0.0, max_value=2.0, value=0.5, step=0.1,
                                help="건축법 시행령 제86조 제3항 제2호 나목: 높은 동의 주된 개구부(거실·안방 창)가 낮은 동을 향하면, "
                                     "인동간격을 '10m 이상 + 낮은 동 높이 × 이 배수'로 정합니다 (가목 대신 적용). 남쪽 동을 낮게, 북쪽 동을 높게 배치하면 동 간격이 크게 줄어 용적률을 더 채울 수 있습니다.\n\n"
                                     "· 시행령 최소 0.5배, 실제 수치는 지자체 건축조례로 정합니다.\n"
                                     "· 기본 0.5배: 광명시·시흥시 현행 조례 모두 시행령 최소 기준(10m 이상 + 낮은 동 높이의 0.5배)을 그대로 적용. 다른 지자체는 조례를 확인해 바꾸세요.\n"
                                     "· 0이면 적용하지 않고 모든 경우 인동간격(H 배수)만 씁니다.")
    min_ns_dist = 10.0
    setback_x = 3.0
    setback_y = 3.0

    with st.expander("적용 규제 요약", expanded=False):
        st.markdown(f"""
- **일조 사선 ({sunlight_dir}):** {'북쪽' if sunlight_dir == '정북방향' else '남쪽'} 인접대지경계선에서 H/2 이상 (도로·공원은 중심선 기준)
- **채광창 이격:** 창이 있는 벽면에서 직각 방향으로 인접대지경계선까지 0.5H 이상
- **인동간격 (가목):** {h_multiplier:g}H 이상, 최소 {min_ns_dist:g} m (두 동 중 높은 쪽 기준)
- **남저북고 완화 (나목):** {('높은 동의 창이 낮은 동을 향하면 10 m 이상 + 낮은 동 높이 × ' + format(low_ratio, 'g')) if low_ratio > 0 else '적용 안 함'}
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
        flip_h, flip_v, str(eff), min_tower_pct, low_ratio
    )

    def run_layout(groups, floors_limit=None):
        # 탑상형 최소 비율은 판상과 탑상이 섞이는 안에만 의미가 있음 (나머지는 0으로 캐시 공유)
        mixes = "판상" in groups and any(g != "판상" for g in groups)
        return auto_optimize_layout(
            site_w, site_l, site_shape_type, floors if floors_limit is None else floors_limit, h_multiplier, setback_x, setback_y, min_ns_dist, side_dist, max_far, max_bcr, selected_sizes, size_ratios, school_n, school_s, school_e, school_w, road_n, road_s, road_e, road_w, sunlight_dir, trap_bottom=trap_bottom, trap_top=trap_top, trap_height=trap_height, l_w=l_w, l_l=l_l, l_w_inner=l_w_inner, l_l_inner=l_l_inner, eff=eff, flip_h=flip_h, flip_v=flip_v, allowed_groups=tuple(groups), min_floors=min_floors,
            min_tower_ratio=min_tower_ratio if mixes else 0.0, low_ratio=low_ratio)

    def count_units(blds):
        return sum(b[5] * b[4] for b in blds)

    def far_of(blds, s_area):
        """용적률(%) = 공급면적 합 ÷ 대지면적"""
        return sum(type_area_breakdown(b[7], eff)[1] * b[4] for b in blds) / s_area * 100 if s_area > 0 else 0.0

    calc_btn = st.button("🚀 배치 계산", type="primary", width="stretch")

    if calc_btn:
        st.session_state['last_inputs'] = inputs_tuple
        st.session_state.pop('far_search', None)
        with st.spinner("판상형·탑상형·혼합 3가지 안을 계산하고 있습니다... (대지 크기·학교 조건에 따라 수십 초\\~2분)"):

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

            bldgs, s_area, b_area, base_site_poly = combo_results[plan_key]
            current_units = count_units(bldgs)
            current_comp = describe_composition(bldgs)
            cur_share = tower_share(bldgs)
            head = f"**{plan_key}안:** {current_comp} · {current_units:,}세대"
            if min_tower_ratio > 0 and current_units > 0 and cur_share < min_tower_ratio - RATIO_TOL:
                if any(meets.values()):
                    tail = f"⚠️ 탑상형 {cur_share:.0%}로 기준 {min_tower_pct}%에 미달합니다 (참고용). 기준을 충족하는 최대안은 **{best_style}안 {best_units:,}세대**입니다."
                else:
                    tail = f"⚠️ 탑상형 {cur_share:.0%}로 기준 {min_tower_pct}%에 미달하고, 기준을 충족하는 안이 없습니다."
                hint = ("warning", f"{head}\n\n{tail}\n\n{compare_txt}")
            elif best_units > current_units and not is_near_tie(best_units, current_units):
                hint = ("warning", f"{head}\n\n💡 **{best_style}안**({best_comp})이 **{best_units:,}세대 (+{best_units - current_units:,})**로 더 많습니다. '{best_style}'을 고르면 그 안의 배치도를 볼 수 있습니다.\n\n{compare_txt}")
            elif best_units > current_units:
                hint = ("info", f"{head}\n\n최대안({best_style}안 {best_units:,}세대)과 차이가 {NEAR_TIE_RATIO:.0%} 이내라 사실상 최대입니다.\n\n{compare_txt}")
            else:
                scope = "탑상형 기준을 충족하는 안 중" if min_tower_ratio > 0 and any(meets.values()) else "비교한 안 중"
                hint = ("success", f"{head}\n\n✅ {scope} 세대수가 가장 많습니다.\n\n{compare_txt}")

            st.session_state['sim_result'] = (bldgs, s_area, b_area, base_site_poly)
            st.session_state['hint'] = hint
            st.session_state['shown_plan'] = plan_key

    if 'last_inputs' in st.session_state and st.session_state['last_inputs'] != inputs_tuple:
        st.warning("입력값이 바뀌었습니다. '🚀 배치 계산'을 다시 눌러주세요.")
        st.stop()

    if 'sim_result' not in st.session_state:
        st.info("왼쪽에서 조건을 입력한 뒤 '🚀 배치 계산'을 누르세요. 처음 계산은 수십 초 걸릴 수 있습니다.")
        st.stop()

    buildings, site_area, bldg_area, base_site_poly = st.session_state['sim_result']

    result_box = st.container()
    far_box = st.container()
    plot_box = st.container()
    shadow_box = st.container()
    override_box = st.container()
    area_box = st.container()

    # ------------------------------------------------------------------
    # 용적률 상한 달성 층수 역산: 층수 제한만 올려 가며 다시 계산
    # ------------------------------------------------------------------
    FAR_SEARCH_MAX_FLOORS = 100     # 층수 입력 상한과 동일
    FAR_REACH_RATIO = 0.98          # 상한의 98% 이상이면 달성으로 봄 (동 단위 배치라 딱 맞추기 어려움)
    with far_box:
        plan_now = st.session_state.get('shown_plan', '탑상형')
        with st.expander(f"🎯 용적률 상한({max_far}%) 달성에 필요한 층수 찾기", expanded='far_search' in st.session_state):
            st.caption(f"다른 조건은 그대로 두고 **최고 층수만 올려 가며** '{plan_now}안'을 다시 계산해, 용적률 상한에 닿는 최저 층수를 찾습니다 "
                       f"(동 단위로 배치해 딱 맞추기 어려우므로 상한의 {FAR_REACH_RATIO:.0%} 이상이면 달성으로 봅니다). "
                       f"여러 번 계산하므로 수 분 걸릴 수 있습니다. 결과는 시뮬레이터 기준이며, 실제 층수 상향 가능 여부는 지구단위계획·고도제한으로 확인하세요.")
            if st.button("🎯 필요 층수 계산", key="far_search_btn"):
                groups = AUTO_COMBOS[plan_now]
                target = max_far * FAR_REACH_RATIO
                tried = {}
                prog = st.progress(0.0, text="계산 준비 중...")

                def evaluate(fl):
                    if fl not in tried:
                        prog.progress(min(0.95, len(tried) / 10.0), text=f"{fl}층 제한으로 계산 중... ({len(tried) + 1}번째)")
                        res = run_layout(groups, floors_limit=fl)
                        tried[fl] = (far_of(res[0], res[1]), count_units(res[0]))
                    return tried[fl][0]

                f_start = int(floors)
                answer, lo, status = None, f_start, ""
                if evaluate(f_start) >= target:
                    answer, status = f_start, "already"
                else:
                    # 5층씩 올려 상한에 닿는 구간을 찾고, 용적률이 더 오르지 않으면(3번 연속 +1%p 미만) 한계로 판단
                    # (판상형 열 배치는 층수가 열 간격과 맞물려 몇 단계 정체 후 뛰는 경우가 있어 3번까지 봄)
                    best_far, stall, f = tried[f_start][0], 0, f_start
                    while f < FAR_SEARCH_MAX_FLOORS:
                        f = min(FAR_SEARCH_MAX_FLOORS, f + 5)
                        far_f = evaluate(f)
                        if far_f >= target:
                            answer = f
                            break
                        lo = f
                        if far_f > best_far + 1.0:
                            best_far, stall = far_f, 0
                        else:
                            stall += 1
                            if stall >= 3:
                                break
                    if answer is not None:
                        # 직전 미달 층수와 달성 층수 사이를 이진 탐색해 최저 층수 확정
                        hi = answer
                        while hi - lo > 1:
                            mid = (lo + hi) // 2
                            if evaluate(mid) >= target:
                                hi = mid
                            else:
                                lo = mid
                        answer, status = hi, "found"
                    else:
                        status = "limit"
                prog.empty()
                st.session_state['far_search'] = {"plan": plan_now, "max_far": max_far, "start": f_start,
                                                  "answer": answer, "status": status, "tried": tried}

            fs = st.session_state.get('far_search')
            if fs:
                best_fl = max(fs["tried"], key=lambda k: fs["tried"][k][0])
                if fs["status"] == "already":
                    st.success(f"현재 최고 층수 {fs['start']}층으로 이미 용적률 상한({fs['max_far']}%)에 닿습니다.")
                elif fs["status"] == "found":
                    far_a, units_a = fs["tried"][fs["answer"]]
                    st.success(f"**{fs['plan']}안은 최고 층수 {fs['answer']}층**이면 용적률 상한 {fs['max_far']}%에 닿습니다 "
                               f"(용적률 {far_a:.1f}%, {units_a:,}세대). 현재 {fs['start']}층 대비 **+{fs['answer'] - fs['start']}층**.\n\n"
                               f"이 배치를 보려면 왼쪽 '최고 층수'를 {fs['answer']}층으로 바꾸고 다시 계산하세요.")
                else:
                    far_b, units_b = fs["tried"][best_fl]
                    st.warning(f"**층수를 올려도 용적률 상한 {fs['max_far']}%에 닿지 않습니다.** {fs['plan']}안의 최대는 "
                               f"{best_fl}층 제한에서 용적률 {far_b:.1f}% ({units_b:,}세대)입니다. "
                               f"층수가 높아질수록 인동간격(0.8H)·일조 사선 이격도 커져서, 대지 크기가 한계가 됩니다. "
                               f"건폐율 상한·탑상형 비율·대지 조건을 함께 검토하세요.")
                st.dataframe(pd.DataFrame([{"층수 제한": k, "용적률 (%)": round(v[0], 1), "세대수": v[1]}
                                           for k, v in sorted(fs["tried"].items())]),
                             hide_index=True, width="stretch")

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
        # 비교표에는 모든 안이 나오지만 배치도는 채택(또는 선택)한 안 하나만 그림 → 어느 안인지 명시
        if st.session_state.get('shown_plan'):
            st.markdown(f"🗺️ **배치도: {st.session_state['shown_plan']}안** — {describe_composition(final_buildings)} · {total_units:,}세대  \n"
                        "<span style='color:gray;font-size:0.9em'>다른 안의 배치를 보려면 '4. 동 형태'에서 해당 안을 고르고 다시 계산하세요.</span>",
                        unsafe_allow_html=True)
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
            # 창마다 실제로 적용되는 인동간격으로 음영을 그림: 창 앞 가장 가까운 동과의 관계로 가목/나목 판정
            #  (마주보는 동이 없으면 자기 높이 기준 가목 거리)
            reach = spacing_dist(floors, floors, h_multiplier, 0.0, min_ns_dist) + 5.0
            win_dists = []
            for w in windows:
                front = [ob for ob in final_buildings if ob is not b and _proj(w, reach).intersects(ob[0])]
                if front:
                    near = min(front, key=lambda ob: LineString([(w[0], w[1]), (w[2], w[3])]).distance(ob[0]))
                    d = spacing_dist(b_floors, near[4], h_multiplier, low_ratio, min_ns_dist)
                    rule = "남저북고 완화(나목)" if (low_ratio > 0 and b_floors > near[4]) else "가목"
                else:
                    d, rule = spacing_dist(b_floors, b_floors, h_multiplier, 0.0, min_ns_dist), "가목"
                win_dists.append((w, d, rule))
            dist_txt = " / ".join(sorted({f"{d:.0f}m ({rule})" for _, d, rule in win_dists}))
            hover_text = f"{idx+1}동 · {type_label(name)}<br>{b_floors}층 · 층당 {b_units}세대 (동 전체 {b_units * b_floors}세대)<br>창 앞 인동간격 {dist_txt}"
            fig.add_trace(go.Scatter(x=list(x), y=list(y), fill='toself', fillcolor=get_type_color(name), mode='lines', line=dict(color='#333333', width=1), text=hover_text, hoverinfo='text', showlegend=False))
            # 세대 구분선은 그리지 않음 (동 외곽만 표시, 세대 구성은 hover로 확인)

            for ((wx1, wy1, wx2, wy2, dx, dy), d_w, _) in win_dists:
                proj_poly = Polygon([(wx1, wy1), (wx2, wy2), (wx2 + dx * d_w, wy2 + dy * d_w), (wx1 + dx * d_w, wy1 + dy * d_w)])
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
