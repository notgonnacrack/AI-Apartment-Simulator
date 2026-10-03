import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
import math
from shapely.geometry import Polygon, box, LineString
from shapely.affinity import translate, rotate as shapely_rotate
from shapely.ops import unary_union

# 한글 폰트 깨짐 방지 (윈도우 맑은 고딕)
import platform
import os
import matplotlib.font_manager as fm

font_path = "NanumGothic.ttf"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rc('font', family='NanumGothic')
elif platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

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
    "55m²(계단식)": (*plate_dims(55, 2, 10.5), 2, 55.0, "판상형"),
    "59m²(계단식)": (*plate_dims(59, 2, 11.0), 2, 59.0, "판상형"),
    "65m²(계단식)": (*plate_dims(65, 2, 11.0), 2, 65.0, "판상형"),
    "74m²(계단식)": (*plate_dims(74, 2, 11.5), 2, 74.0, "판상형"),
    "84m²(계단식)": (*plate_dims(84, 2, 12.0), 2, 84.0, "판상형"),
    "84m²(탑상형)": (*tower_dims(84), 3, 84.0, "L자형"),
    "114m²(계단식)": (*plate_dims(114, 2, 13.0), 2, 114.0, "판상형"),
    "114m²(타워형)": (*tower_dims(114), 3, 114.0, "L자형"),
}

SIZE_COLORS = {
    "84m²(탑상형)": "#3cb371",
    "84m²": "#6495ed",
    "114m²": "#ff69b4",
    "59m²": "#00ced1",
    "74m²": "#ff7f50",
    "65m²": "#dda0dd",
    "55m²": "#ffd700",
    "46m²": "#4682b4",
    "41m²": "#ff69b4",
    "36m²": "#cd5c5c",
    "31m²": "#8fbc8f",
    "26m²": "#9370db",
    # 복합 동 타입 (탑상 4호 / 판탑 혼합동)
    "84m²(탑상4호)": "#2e8b57",
    "84m²(판탑2+2)": "#f4a460",
    "84m²(판탑2+3)": "#d2691e",
    "114m²(탑상4호)": "#556b2f",
    "114m²(판탑2+2)": "#bc8f8f",
    "114m²(판탑2+3)": "#a0522d",
}

def get_type_color(name):
    if name in SIZE_COLORS:
        return SIZE_COLORS[name]
    # 기타 평형 등으로 자동 생성된 복합 타입의 기본 색상
    if "탑상4호" in name: return "#2e8b57"
    if "판탑2+2" in name: return "#f4a460"
    if "판탑2+3" in name: return "#d2691e"
    return SIZE_COLORS.get(name.split('(')[0], '#cccccc')

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
        t = b_w / 3.0
        dividers.append(((t, 0), (t, t)))
        dividers.append(((0, t), (t, t)))
        if units >= 4:
            dividers.append(((t*2, 0), (t*2, t)))
        if units >= 5:
            dividers.append(((0, t*2), (t, t*2)))
    return dividers

def create_building_poly(b_w, b_l, bldg_shape):
    if bldg_shape in ["타워형", "L자형"]:
        t = b_w / 3.0
        return Polygon([(0,0), (b_w,0), (b_w, t), (t, t), (t, b_l), (0, b_l)])
    else:
        return box(0, 0, b_w, b_l)

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
# 복합 동 타입 자동 생성 (탑상 4호 / 판탑 혼합동)
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

COMPOSITE_GEOM = {}  # name -> (poly, windows, dividers)

def build_composite_types(unit_types, composite_geom):
    by_size = {}
    for name, spec in list(unit_types.items()):
        size_label = name.split("(")[0]
        shape = spec[4]
        if shape == "판상형" and spec[2] == 2:
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
        ym = t + (W - t) / 2.0
        div4 = [((W / 2.0, 0), (W / 2.0, t)), ((0, t), (t, t)), ((0, ym), (t, ym))]
        n4 = f"{size_label}(탑상4호)"
        unit_types[n4] = (W, W, 4, area, "탑상4호")
        composite_geom[n4] = (poly4, win4, div4)

        # 2) 판탑 혼합동 : 판상 2세대(남측 날개) + 탑상 2~3세대(꺾인 날개)
        for n_tower in (2, 3):
            LA = 2 * pw
            LB = pd + n_tower * u_t
            poly_m = Polygon([(0, 0), (LA, 0), (LA, pd), (t, pd), (t, LB), (0, LB)])
            win_m = [(0, 0, pw, 0, 0, -1), (pw, 0, LA, 0, 0, -1), (0, pd, 0, LB, -1, 0)]
            div_m = [((pw, 0), (pw, pd)), ((0, pd), (t, pd))]
            for k in range(1, n_tower):
                yk = pd + k * u_t
                div_m.append(((0, yk), (t, yk)))
            nm = f"{size_label}(판탑2+{n_tower})"
            unit_types[nm] = (LA, LB, 2 + n_tower, area, "판탑형")
            composite_geom[nm] = (poly_m, win_m, div_m)

build_composite_types(UNIT_TYPES, COMPOSITE_GEOM)

def get_type_group(name, shape, units):
    if shape == "판상형": return "판상"
    if shape == "L자형": return "탑상3"
    if shape == "탑상4호": return "탑상4"
    if shape == "판탑형": return f"판탑{units}"
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
    elif shape == "판탑형":
        supply = area / e_stair * 2 + area / e_tower * (units - 2)
    else:
        supply = area / e_stair * units
    return area * units, supply

ALL_GROUPS = ("판상", "탑상3", "탑상4", "판탑4", "판탑5")
# AI 자동 모드에서 비교하는 조합들
AUTO_COMBOS = {
    "판상형만": ("판상",),
    "탑상형 (3호·4호)": ("탑상3", "탑상4"),
    "판탑 혼합동 + 판상형": ("판상", "판탑4", "판탑5"),
    "전체 혼합 (판상+탑상+판탑)": ALL_GROUPS,
}

@st.cache_data(show_spinner=False)
def auto_optimize_layout(site_w, site_l, site_shape_type, floors, h_multiplier, setback_x, setback_y, min_ns_dist, side_dist, max_far, max_bcr, selected_sizes, size_ratios, school_n, school_s, school_e, school_w, road_n, road_s, road_e, road_w, sunlight_dir, trap_bottom=0, trap_top=0, trap_height=0, l_w=0, l_l=0, l_w_inner=0, l_l_inner=0, eff=DEFAULT_EFF, layout_version=20, flip_h=False, flip_v=False, plate_only=False, tower_only=False, allowed_groups=None):
    
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
    
    # ---- 사선/채광 이격 판정용 '인접대지경계선' 영역 (실제 대지 경계 기준) ----
    # 도로·공원 등이 있으면 공동주택은 그 '중심선'을 인접대지경계선으로 봄 (건축법 시행령 제86조 제6항)
    # → 해당 방향으로 대지를 (빈 땅 너비 / 2) 만큼 밀어낸 영역까지 허용
    from shapely.prepared import prep
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
    
    school_polys = []
    off_n = max(5, road_n)
    if school_n: school_polys.append(box(minx, maxy + off_n, maxx, maxy + off_n + 100))
    off_s = max(5, road_s)
    if school_s: school_polys.append(box(minx, miny - off_s - 100, maxx, miny - off_s))
    off_e = max(5, road_e)
    if school_e: school_polys.append(box(maxx + off_e, miny, maxx + off_e + 100, maxy))
    off_w = max(5, road_w)
    if school_w: school_polys.append(box(minx - off_w - 100, miny, minx - off_w, maxy))
    
    buildings = []
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
        if name in COMPOSITE_GEOM:
            poly, windows, dividers = COMPOSITE_GEOM[name]
        else:
            poly = create_building_poly(w, l, shape)
            windows = get_base_windows(w, l, shape, units)
            dividers = get_base_dividers(w, l, shape, units)
        _excl_floor, supply_floor = type_area_breakdown(name, eff)
        types_info.append({
            'name': name, 'size_label': size_label, 'poly': poly, 'units': units, 'shape': shape,
            'area': poly.area, 'windows': windows, 'dividers': dividers,
            'supply_floor': supply_floor  # 층당 공급면적 합 (용적률 산정용)
        })
    
    points = []
    # 대지가 커지면 연산 속도를 위해 탐색 간격을 넓힘
    step_size = max(15.0, min(maxx - minx, maxy - miny) / 30.0)
    for x in np.arange(minx + setback_x, maxx - setback_x, step_size):
        for y in np.arange(miny + setback_y, maxy - setback_y, step_size):
            points.append((x,y))
            
    random.seed(42)
    random.shuffle(points)
    
    total_floor_area = 0
    total_units = 0
    bldg_area = 0
    placed_counts = {size: 0 for size in selected_sizes}
    
    for (x, y) in points:
        placed = False
        
        total_ratio = sum(size_ratios.values())
        total_u = sum(placed_counts.values())
        
        def get_priority(t):
            target_pct = size_ratios[t['size_label']] / total_ratio if total_ratio > 0 else 0
            current_pct = placed_counts[t['size_label']] / total_u if total_u > 0 else 0
            deficit = target_pct - current_pct
            density = t['units'] / t['area']
            return (deficit, density)
            
        current_types = sorted(types_info, key=get_priority, reverse=True)
        
        for t_info in current_types:
            if placed: break
            
            start_floor = random.randint(max(10, floors - 10), floors)
            # 그림자 제약(학교)을 피하기 위해 층수를 깎아내려가며(Step-down) 테스트
            for b_floors in range(start_floor, 4, -3):
                if placed: break
                
                b_h = b_floors * 3.0
                b_ns_dist = max(min_ns_dist, b_h * h_multiplier)
                
                # 용적률: 지상 연면적 = 세대 공급면적(전용+주거공용) 합계 / 건폐율: 동 외곽 바닥면적(건축면적)
                new_far = ((total_floor_area + t_info['supply_floor'] * b_floors) / site_area) * 100
                new_bcr = ((bldg_area + t_info['area']) / site_area) * 100
                
                if new_bcr > max_bcr:
                    break # 이 형태는 건폐율 초과라 층수를 깎아도 해결 불가. 다음 형태로.
                if new_far > max_far:
                    continue # 층수를 깎으면 해결될 수 있으므로 계속 시도.
                    
                rotations = [0, 45, 315]  # 남향, 남동향, 남서향만 검토 (속도 최적화 및 탑상형 지원)
                random.shuffle(rotations)
                
                for angle in rotations:
                    final_windows = transform_windows(t_info['windows'], angle, x, y)
                    if not is_valid_orientation(final_windows):
                        continue
                        
                    candidate = shapely_rotate(t_info['poly'], angle, origin=(0,0))
                    candidate = translate(candidate, xoff=x, yoff=y)
                    if not site_poly.contains(candidate):
                        continue
                    
                    conflict = False
                    
                    # 0. 학교 일조권 보호 (그림자 침범 검사) - 최적화됨
                    if school_polys:
                        # bounding box of candidate
                        cx_min, cy_min, cx_max, cy_max = candidate.bounds
                        for t_hr in [9.0, 12.0, 15.0]:
                            if conflict: break
                            az_deg, alt_deg = get_solar_angle(t_hr)
                            s_len = b_h / math.tan(math.radians(alt_deg))
                            sdx = math.sin(math.radians(az_deg - 180)) * s_len
                            sdy = math.cos(math.radians(az_deg - 180)) * s_len
                            
                            # Fast AABB check
                            s_minx = min(cx_min, cx_min + sdx)
                            s_maxx = max(cx_max, cx_max + sdx)
                            s_miny = min(cy_min, cy_min + sdy)
                            s_maxy = max(cy_max, cy_max + sdy)
                            
                            for spoly in school_polys:
                                sp_minx, sp_miny, sp_maxx, sp_maxy = spoly.bounds
                                # If bounding boxes don't intersect, skip heavy math
                                if s_maxx < sp_minx or s_minx > sp_maxx or s_maxy < sp_miny or s_miny > sp_maxy:
                                    continue
                                
                                shift_c = translate(candidate, xoff=sdx, yoff=sdy)
                                shadow_c = unary_union([candidate, shift_c]).convex_hull
                                if shadow_c.intersects(spoly):
                                    conflict = True
                                    break
                                
                    # 0-1. 일조 사선제한 (건축법 시행령 제86조 제1항: 높이 10m 초과 부분은 높이의 1/2 이상)
                    #      건물을 H/2 만큼 정북(정남 기준이면 정남)으로 밀어도 인접대지경계선 안에 있어야 함
                    #      → 사다리꼴 빗변, L자 파인 부분 등 실제 경계 전체에 대해 판정
                    if not conflict:
                        sun_dy = (b_h / 2.0) if sunlight_dir == "정북방향" else -(b_h / 2.0)
                        if not sun_region_p.contains(translate(candidate, yoff=sun_dy)):
                            conflict = True
                            
                    # 0-2. 채광창 방향 이격 (시행령 제86조 제3항 제1호: 창이 있는 벽면에서 '직각 방향'으로 0.5H 이상)
                    if not conflict:
                        window_setback = b_h * 0.5
                        for (wx1, wy1, wx2, wy2, dx, dy) in final_windows:
                            win_proj = Polygon([(wx1, wy1), (wx2, wy2),
                                                (wx2 + dx * window_setback, wy2 + dy * window_setback),
                                                (wx1 + dx * window_setback, wy1 + dy * window_setback)])
                            if not win_region_p.contains(win_proj):
                                conflict = True
                                break

                    if conflict: continue
                    
                    c_minx, c_miny, c_maxx, c_maxy = candidate.bounds
                    for existing_bldg, _, ex_windows, _, ex_floors, _, ex_ns_dist, _ in buildings:
                        e_minx, e_miny, e_maxx, e_maxy = existing_bldg.bounds
                        ex_h = ex_floors * 3.0
                        req_ns_dist = max(min_ns_dist, max(b_h, ex_h) * h_multiplier)
                        max_possible_dist = req_ns_dist + 5.0
                        
                        if c_maxx < e_minx - max_possible_dist or c_minx > e_maxx + max_possible_dist or c_maxy < e_miny - max_possible_dist or c_miny > e_maxy + max_possible_dist:
                            continue
                        if candidate.intersects(existing_bldg.buffer(side_dist, resolution=2)):
                            conflict = True
                            break
                        
                        for (wx1, wy1, wx2, wy2, dx, dy) in final_windows:
                            proj_poly = Polygon([(wx1, wy1), (wx2, wy2), (wx2 + dx * req_ns_dist, wy2 + dy * req_ns_dist), (wx1 + dx * req_ns_dist, wy1 + dy * req_ns_dist)])
                            if proj_poly.intersects(existing_bldg):
                                conflict = True
                                break
                        if conflict: break
                        
                        for (wx1, wy1, wx2, wy2, dx, dy) in ex_windows:
                            proj_poly = Polygon([(wx1, wy1), (wx2, wy2), (wx2 + dx * req_ns_dist, wy2 + dy * req_ns_dist), (wx1 + dx * req_ns_dist, wy1 + dy * req_ns_dist)])
                            if proj_poly.intersects(candidate):
                                conflict = True
                                break
                        if conflict: break
                            
                    if not conflict:
                        final_dividers = transform_dividers(t_info['dividers'], angle, x, y)
                        buildings.append((candidate, t_info['shape'], final_windows, final_dividers, b_floors, t_info['units'], b_ns_dist, t_info['name']))
                        total_floor_area += t_info['supply_floor'] * b_floors
                        bldg_area += t_info['area']
                        added_units = t_info['units'] * b_floors
                        total_units += added_units
                        placed_counts[t_info['size_label']] += added_units
                        placed = True
                        break
                    
    return buildings, site_area, bldg_area, base_site_poly

st.set_page_config(layout="wide", page_title="속 터져서 내가 직접 만들어본 공동주택 가배치")

import os
@st.dialog("📖 AI 아파트 가배치 시뮬레이터 사용 메뉴얼", width="large")
def show_manual():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    manual_path = os.path.join(current_dir, "app_manual.md")
    if os.path.exists(manual_path):
        with open(manual_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.error("매뉴얼 파일을 찾을 수 없습니다.")

col_title, col_btn = st.columns([5, 1])
with col_title:
    st.title("속 터져서 내가 직접 만들어 본 공동주택 假배치 😤")
with col_btn:
    st.write("")
    st.write("")
    if st.button("📖 사용 매뉴얼 띄우기", use_container_width=True, type="secondary"):
        show_manual()
st.markdown("<div style='background-color: #ffe066; padding: 5px 15px; border-radius: 5px; display: inline-block; font-weight: 800; font-size: 1.1em; color: #333333; margin-bottom: 20px;'>© 2026 김진우 (Jinwoo Kim). All rights reserved.</div>", unsafe_allow_html=True)

col_input, col_viz = st.columns([1, 2])

with col_input:
    st.header("1. 대지 정보 및 주변 환경 설정")
    site_shape_type = st.selectbox("대지 형상 (Site Shape)", ["직사각형", "L자형", "ㄱ자형", "사다리꼴"])
    
    # 기본값 초기화
    trap_bottom, trap_top, trap_height = 0, 0, 0
    l_w, l_l, l_w_inner, l_l_inner = 0, 0, 0, 0
    
    if site_shape_type == "직사각형":
        st.caption("※ 선택한 형상에 맞게 가로/세로 길이로 다각형 대지가 생성됩니다.")
        site_w = st.number_input("대지 가로 길이 (m)", min_value=30, max_value=500, value=200, step=10)
        site_l = st.number_input("대지 세로 길이 (m)", min_value=30, max_value=500, value=200, step=10)
    elif site_shape_type == "사다리꼴":
        st.caption("※ 사다리꼴의 치수를 상세 입력합니다.")
        trap_bottom = st.number_input("아랫변 길이 (긴변, m)", min_value=30, max_value=500, value=250, step=10)
        trap_top = st.number_input("윗변 길이 (짧은변, m)", min_value=10, max_value=500, value=150, step=10)
        trap_height = st.number_input("높이 (m)", min_value=30, max_value=500, value=200, step=10)
        site_w = max(trap_bottom, trap_top)
        site_l = trap_height
    elif site_shape_type == "L자형" or site_shape_type == "ㄱ자형":
        st.caption("※ 선택하신 대지의 치수를 상세 입력합니다.")
        col_w1, col_w2 = st.columns(2)
        l_w = col_w1.number_input("가로 전체 길이 (m)", min_value=30, max_value=500, value=250, step=10)
        l_w_inner = col_w2.number_input("파인 부분 가로 (m)", min_value=10, max_value=500, value=120, step=10)
        col_l1, col_l2 = st.columns(2)
        l_l = col_l1.number_input("세로 전체 길이 (m)", min_value=30, max_value=500, value=200, step=10)
        l_l_inner = col_l2.number_input("파인 부분 세로 (m)", min_value=10, max_value=500, value=100, step=10)
        # 예외 처리 (파인 부분이 전체보다 크지 않게)
        l_w_inner = min(l_w_inner, l_w - 10)
        l_l_inner = min(l_l_inner, l_l - 10)
        site_w = l_w
        site_l = l_l

    flip_h = False
    flip_v = False
    if site_shape_type in ["L자형", "ㄱ자형", "사다리꼴"]:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            flip_h = st.checkbox("좌우 반전 (대칭)", value=False)
        with col_f2:
            flip_v = st.checkbox("상하 반전 (대칭)", value=False)
    max_far = st.number_input("용적률 상한 (%)", min_value=50, max_value=1000, value=300, step=10)
    max_bcr = st.number_input("건폐율 상한 (%)", min_value=10, max_value=100, value=20, step=2, help="💡 [수익성 팁] 법정 최대 건폐율(예: 60%)을 꽉 채우면 일조권 사선제한 때문에 오히려 층수가 깎입니다. 20~25% 수준으로 넉넉히 비워야 건물을 높게 올려 최대 세대수를 뽑을 수 있습니다.")
    with st.expander("📐 면적 산정 기준 (전용률 · 기타공용)", expanded=False):
        st.caption("용적률은 법령대로 **세대별 공급면적(전용 + 주거공용) 합계**로 산정합니다. 발코니(서비스면적)는 바닥면적에서 제외되므로 용적률에 들어가지 않습니다.")
        c_e1, c_e2, c_e3 = st.columns(3)
        eff_stair = c_e1.number_input("계단식 전용률 (%)", min_value=50, max_value=95, value=76, step=1, help="공급면적 대비 전용면적 비율. 84㎡ 기준 76% → 공급 약 110.5㎡")
        eff_tower = c_e2.number_input("탑상형 전용률 (%)", min_value=50, max_value=95, value=74, step=1, help="코어·복도가 커서 계단식보다 약간 낮음. 84㎡ 기준 74% → 공급 약 113.5㎡")
        eff_corr = c_e3.number_input("복도식 전용률 (%)", min_value=50, max_value=95, value=72, step=1, help="공용복도 때문에 가장 낮음")
        other_common_pct = st.number_input("기타공용면적 (공급면적 대비 %)", min_value=0, max_value=150, value=50, step=5, help="지하주차장·관리동·커뮤니티 등. 계약면적 = 공급면적 + 기타공용. 용적률에는 들어가지 않으며 집계표 표시용입니다.")
    eff = (eff_stair / 100.0, eff_tower / 100.0, eff_corr / 100.0)
    limit_floors = st.checkbox("층수 제한 있음", value=True)
    if limit_floors:
        floors = st.number_input("최고 층수 제한 (층)", min_value=1, max_value=100, value=35, step=1)
    else:
        floors = 50
        
    st.subheader("일조권 사선제한 방향")
    sunlight_dir = st.radio("적용 기준 (건축법 제61조)", ["정북방향", "정남방향"], horizontal=True, index=0,
                            help="일반적으로 정북방향이 적용되나, 택지개발지구/지구단위계획 등에서는 정남방향이 적용될 수 있습니다.")
        
    st.subheader("교육환경보호구역 (학교 연접 여부)")
    st.caption("※ 예정지와 학교 사이에 녹지나 도로가 있다면 학교에 체크하고 아래 이격거리에 도로나 공원 폭을 입력하세요")
    col_sch_n, col_sch_s = st.columns(2)
    school_n = col_sch_n.checkbox("북쪽 학교 (그림자 검사)", value=False)
    school_s = col_sch_s.checkbox("남쪽 학교 (그림자 검사)", value=False)
    col_sch_e, col_sch_w = st.columns(2)
    school_e = col_sch_e.checkbox("동쪽 학교 (그림자 검사)", value=False)
    school_w = col_sch_w.checkbox("서쪽 학교 (그림자 검사)", value=False)
    
    st.subheader("인접 사유지와의 이격거리 (도로/공원 너비)")
    st.caption("대지경계선 밖에 빈 땅(도로, 공원)이 있다면 너비(m)를 입력하세요. (0m = 사유지와 바로 맞닿음)")
    col_n, col_s = st.columns(2)
    road_n = col_n.number_input("북쪽 빈 땅 너비 (m)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
    road_s = col_s.number_input("남쪽 빈 땅 너비 (m)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
    col_e, col_w = st.columns(2)
    road_e = col_e.number_input("동쪽 빈 땅 너비 (m)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
    road_w = col_w.number_input("서쪽 빈 땅 너비 (m)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
    
    st.header("2. 평형 선택 (전용면적 기준)")
    st.caption("배치에 사용할 평형을 모두 선택해주세요")
    STYLE_OPTIONS = {
        "🤖 AI 자동 (세대수 최대 조합)": None,
        "판상형만": ("판상",),
        "탑상형만 (3호·4호)": ("탑상3", "탑상4"),
        "판탑 혼합동만": ("판탑4", "판탑5"),
    }
    layout_style = st.radio("배치 스타일 최적화 (건물 형태)", 
                           list(STYLE_OPTIONS.keys()), 
                           horizontal=True,
                           help="AI 자동: 판상형 / 탑상형(3호·4호) / 판탑 혼합동 / 전체 혼합 4가지 조합을 모두 돌려보고 세대수가 가장 많은 조합을 채택합니다.\n\n※ 판탑 혼합동·탑상 4호는 판상형과 탑상형이 모두 있는 평형(84m², 114m², 기타 평형)에서만 생성됩니다.")
    style_groups = STYLE_OPTIONS[layout_style]
    is_auto_style = style_groups is None
    
    st.write("초소형(복도식)")
    c1, c2, c3, c4, c5 = st.columns(5)
    use_26 = c1.checkbox("26m²", value=False)
    use_31 = c2.checkbox("31m²", value=False)
    use_36 = c3.checkbox("36m²", value=False)
    use_41 = c4.checkbox("41m²", value=False)
    use_46 = c5.checkbox("46m²", value=False)
    st.write("소/중대형(계단식)")
    c6, c7, c8, c9, c10, c11 = st.columns(6)
    use_55 = c6.checkbox("55m²", value=False)
    use_59 = c7.checkbox("59m²", value=False)
    use_65 = c8.checkbox("65m²", value=False)
    use_74 = c9.checkbox("74m²", value=False)
    use_84 = c10.checkbox("84m²", value=True)
    use_114 = c11.checkbox("114m²", value=False)
    
    st.write("기타 평형 (직접 입력)")
    use_custom = st.checkbox("기타 평형 활성화", value=False)
    if use_custom:
        c_area = st.number_input("전용면적 (m²) - 숫자만 입력하시면 판상형/타워형을 AI가 자동으로 모두 생성합니다.", min_value=10, max_value=300, value=135)
        
        import math
        # 1. 판상형 자동 생성 (2세대 기준)
        p_footprint = 2.85 * c_area * 2
        p_b_l = round(math.sqrt(p_footprint / 2.0), 1)
        p_b_w = round(p_b_l * 2.0, 1)
        UNIT_TYPES[f"{c_area}m²(판상형)"] = (p_b_w, p_b_l, 2, float(c_area), "판상형")
        
        # 2. 타워형 자동 생성 (3세대 기준)
        t_footprint = 2.85 * c_area * 3
        t_b_w = round(math.sqrt(t_footprint * 9 / 5.0), 1)
        UNIT_TYPES[f"{c_area}m²(타워형)"] = (t_b_w, t_b_w, 3, float(c_area), "L자형")
        
        # 3. 탑상 4호 / 판탑 혼합동도 자동 생성
        build_composite_types(UNIT_TYPES, COMPOSITE_GEOM)

    selected_sizes = []
    if use_26: selected_sizes.append("26m²")
    if use_31: selected_sizes.append("31m²")
    if use_36: selected_sizes.append("36m²")
    if use_41: selected_sizes.append("41m²")
    if use_46: selected_sizes.append("46m²")
    if use_55: selected_sizes.append("55m²")
    if use_59: selected_sizes.append("59m²")
    if use_65: selected_sizes.append("65m²")
    if use_74: selected_sizes.append("74m²")
    if use_84: selected_sizes.append("84m²")
    if use_114: selected_sizes.append("114m²")
    if use_custom: selected_sizes.append(f"{c_area}m²")
    
    if not selected_sizes:
        st.warning("최소 1개 이상의 평형을 선택해주세요.")
        st.stop()
        
    st.subheader("평형별 목표 비율 (%)")
    st.caption("※ 선택한 평형 비율의 합이 반드시 100%가 되어야 시뮬레이션이 동작합니다.")
    ratio_cols = st.columns(len(selected_sizes))
    size_ratios = {}
    for i, size in enumerate(selected_sizes):
        default_val = 100 // len(selected_sizes)
        if i == len(selected_sizes) - 1:
            default_val += 100 % len(selected_sizes)
        size_ratios[size] = ratio_cols[i].number_input(f"{size} (%)", min_value=0, max_value=100, value=default_val, step=5)
        
    total_ratio = sum(size_ratios.values())
    if total_ratio != 100:
        st.error(f"비율의 합이 100%가 되어야 합니다. (현재 합계: {total_ratio}%)")
        st.stop()
    
    st.header("3. 건축 규제 세부 설정")
    st.info("※ 건축법 및 지자체 건축조례 기준 적용")
    h_multiplier = st.number_input("인동간격 규정 (H 배수)", min_value=0.1, max_value=2.0, value=0.8, step=0.1, help="광명시 기준 0.8H, 타 지자체는 조례에 따라 다를 수 있습니다.")
    min_ns_dist = 10.0
    side_dist = 4.0
    setback_x = 3.0
    setback_y = 3.0
    
    st.markdown(f"""
    - **인동간격 규정:** {h_multiplier} H (지자체 건축조례)
    - **최소 인동간격:** {min_ns_dist} m
    - **측벽 이격거리:** {side_dist} m (건축법 시행령)
    - **대지경계 이격:** {setback_x} m (대지안의 공지)
    - **{sunlight_dir} 사선제한:** {'북쪽' if sunlight_dir == '정북방향' else '남쪽'} 인접대지경계선으로부터 H/2 이상 이격 (건축법 제61조, 도로·공원은 중심선 기준)
    - **채광창 대지경계 이격:** 창문 방향 인접대지로부터 0.5H 이상 이격
    """)



with col_viz:
    st.info("💡 **교육환경보호 (일조권):** 학교 부지가 설정되면, 오전 9시~오후 3시 사이에 학교로 그림자를 드리우는 건물을 알아서 층수를 깎아(Step-down) 배치합니다.\n\n💡 **대지경계 사선제한:** 주변에 빈 땅(도로)이 있을 경우 그 너비만큼 일조권(H/2) 및 이격(0.5H) 제한을 완화받아 자동 배치됩니다.\n\n⚠️ **참고:** 도면에 그려지는 붉은색 투영 면적은 '건물 간 인동간격(0.8H)' 확인용입니다. 이 면적이 대지경계선 밖(도로 등)으로 튀어나가는 것은 합법입니다.")
    inputs_tuple = (
        layout_style, site_shape_type, trap_bottom, trap_top, trap_height, 
        l_w, l_l, l_w_inner, l_l_inner, max_far, max_bcr,
        limit_floors, floors, sunlight_dir, school_n, school_s, school_e, school_w, 
        road_n, road_s, road_e, road_w, str(selected_sizes), str(size_ratios),
        flip_h, flip_v, str(eff)
    )

    calc_btn = st.button("🚀 시뮬레이션 계산 시작", type="primary", use_container_width=True)
    
    if calc_btn:
        st.session_state['last_inputs'] = inputs_tuple
        with st.spinner("AI가 여러 동 조합(판상/탑상/판탑)을 비교하며 최적 배치를 찾고 있습니다... (약 20~40초 소요)"):
            def run_layout(groups):
                return auto_optimize_layout(
                    site_w, site_l, site_shape_type, floors, h_multiplier, setback_x, setback_y, min_ns_dist, side_dist, max_far, max_bcr, selected_sizes, size_ratios, school_n, school_s, school_e, school_w, road_n, road_s, road_e, road_w, sunlight_dir, trap_bottom=trap_bottom, trap_top=trap_top, trap_height=trap_height, l_w=l_w, l_l=l_l, l_w_inner=l_w_inner, l_l_inner=l_l_inner, eff=eff, flip_h=flip_h, flip_v=flip_v, allowed_groups=tuple(groups))
            
            def count_units(blds):
                return sum([b[5] * b[4] for b in blds])
            
            # 백그라운드 수익성 분석: 모든 조합을 돌려서 세대수 비교
            combo_results = {}
            for combo_name, groups in AUTO_COMBOS.items():
                combo_results[combo_name] = run_layout(groups)
            alternatives = {k: count_units(v[0]) for k, v in combo_results.items()}
            best_style = max(alternatives, key=alternatives.get)
            best_units = alternatives[best_style]
            compare_txt = " · ".join([f"{k} {v:,}세대" for k, v in sorted(alternatives.items(), key=lambda x: -x[1])])
            
            if is_auto_style:
                bldgs, s_area, b_area, base_site_poly = combo_results[best_style]
                hint_msg = f"🤖 **AI 자동 조합 결과:** {len(alternatives)}가지 동 조합을 비교한 결과 **'{best_style}'** 조합이 **{best_units:,}세대**로 가장 많아 채택했습니다.\n\n📊 비교: {compare_txt}"
            else:
                bldgs, s_area, b_area, base_site_poly = run_layout(style_groups)
                current_units = count_units(bldgs)
                if best_units > current_units:
                    diff = best_units - current_units
                    hint_msg = f"💡 **수익성 분석:** '{best_style}' 조합으로 바꾸면 최대 **{best_units:,}세대 (+{diff:,}세대)**까지 뽑을 수 있습니다! ('🤖 AI 자동'을 선택하면 자동 적용됩니다)\n\n📊 비교: {compare_txt}"
                else:
                    hint_msg = f"💡 **수익성 분석:** 완벽합니다! 현재 선택하신 배치가 이 대지에서 뽑아낼 수 있는 **최대 세대수({current_units:,}세대)**입니다!\n\n📊 비교: {compare_txt}"
                    
            st.session_state['sim_result'] = (bldgs, s_area, b_area, base_site_poly)
            st.session_state['hint_msg'] = hint_msg

    if 'last_inputs' in st.session_state and st.session_state['last_inputs'] != inputs_tuple:
        st.warning("⚠️ 입력값이 변경되었습니다. 적용하려면 '🚀 시뮬레이션 계산 시작' 버튼을 다시 눌러주세요.")
        st.stop()
        
    if 'sim_result' not in st.session_state:
        st.info("💡 대지 조건과 법규를 모두 입력하신 후, 위의 '🚀 시뮬레이션 계산 시작' 버튼을 눌러주세요!")
        st.stop()
        
    buildings, site_area, bldg_area, base_site_poly = st.session_state['sim_result']
        
    main_viz = st.container()
    sunlight_ctrls = st.container()
    override_viz = st.container()
    
    with sunlight_ctrls:
        st.subheader("☀️ 일조 시뮬레이션 시간 조절 (배치도 3D 그림자)")
        col_s1, col_s2 = st.columns([1, 2])
        show_shadow = col_s1.checkbox("그림자 표시", value=True)
        time_of_day = col_s2.slider("시간대", min_value=9.0, max_value=15.0, value=12.0, step=0.5, format="%.1f 시")

    with override_viz:
        with st.expander('🏢 동별 층수 수동 조절 (Override) - 필요시 클릭하여 펼치기', expanded=False):
            st.caption("AI가 깎아낸 층수를 무시하고 원하는 층수로 강제 고정할 수 있습니다. (체크 시 그림자/이격거리 무시)")
            
            import pandas as pd
            
            df_data = []
            for i, b in enumerate(buildings):
                df_data.append({
                    "동 이름": f"{i+1}동",
                    "AI 추천 층수": b[4],
                    "강제 층수": b[4],
                    "수동 고정": False
                })
            df = pd.DataFrame(df_data)
            
            edited_df = st.data_editor(
                df,
                column_config={
                    "수동 고정": st.column_config.CheckboxColumn("수동 고정 (Override)", default=False),
                    "강제 층수": st.column_config.NumberColumn("강제 층수", min_value=0, max_value=100, step=1)
                },
                disabled=["동 이름", "AI 추천 층수"],
                hide_index=True,
                use_container_width=True
            )
            
    final_buildings = []
    total_floor_area = 0
    total_units = 0
    actual_bldg_area = 0
    
    # 면적 집계용 (동 타입별)
    sum_exclusive = 0.0
    sum_supply = 0.0
    area_by_type = {}
    
    for i, b in enumerate(buildings):
        poly, shape, windows, dividers, b_floors, units, ns_dist, name = b
        row = edited_df.iloc[i]
        
        # Override logic (빈 칸으로 지운 경우는 AI 추천값 유지)
        if row["수동 고정"] and not pd.isna(row["강제 층수"]):
            b_floors = int(row["강제 층수"])
            
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
    
    with main_viz:
        st.header("배치 및 그림자 간섭 분석")
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        metrics_col1.metric("최대 달성 세대수", f"{total_units:,} 세대")
        metrics_col2.metric("세팅된 동 수", f"{len(final_buildings)} 동")
        metrics_col3.metric("건폐율 (BCR)", f"{bcr:.2f} %")
        metrics_col4.metric("용적률 (FAR)", f"{far:.2f} %")
        
        st.subheader("📊 면적 산정 결과")
        oc = other_common_pct / 100.0
        area_rows = []
        for t_name in sorted(area_by_type):
            a = area_by_type[t_name]
            area_rows.append({
                "동 타입": t_name, "동 수": a["동"], "세대수": a["세대"],
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
        st.dataframe(pd.DataFrame(area_rows), hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(format="%.1f") for c in
                                    ["세대당 공급 (m²)", "전용면적 합계 (m²)", "공급면적 합계 (m²)", "계약면적 합계 (m²)"]})
        st.caption(f"※ 용적률 = 공급면적(전용+주거공용) 합계 ÷ 대지면적. 발코니(서비스면적) 제외. 계약면적 = 공급면적 + 기타공용({other_common_pct}%, 지하주차장·관리동 등 — 용적률 미산입).")

        
        if st.session_state.get('hint_msg'):
            _hint = st.session_state['hint_msg']
            if _hint.startswith('🤖'):
                st.success(_hint.replace('🤖 ', '', 1), icon='🤖')
            else:
                st.warning(_hint.replace('💡 ', ''), icon='💡')
        
        azimuth_deg, altitude_deg = get_solar_angle(time_of_day)
        
        import plotly.graph_objects as go
        fig = go.Figure()
        
        # base_site_poly is already loaded from sim_result
            
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
                b_h = b_floors * 3.0
                shadow_length = b_h / math.tan(math.radians(altitude_deg))
                shadow_dx = math.sin(math.radians(azimuth_deg - 180)) * shadow_length
                shadow_dy = math.cos(math.radians(azimuth_deg - 180)) * shadow_length
                shifted_poly = translate(poly, xoff=shadow_dx, yoff=shadow_dy)
                shadow_poly = unary_union([poly, shifted_poly]).convex_hull
                sx, sy = shadow_poly.exterior.xy
                shadow_x.extend(list(sx) + [None])
                shadow_y.extend(list(sy) + [None])
            fig.add_trace(go.Scatter(x=shadow_x, y=shadow_y, fill='toself', fillcolor='rgba(0,0,0,0.3)', mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False))

        placed_sizes = set([b[7] for b in final_buildings])
        for size in sorted(list(placed_sizes)):
            color = get_type_color(size)
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color=color, size=15, symbol='square'), name=size.replace('계단식', '').replace('()', ''), showlegend=True))

        for idx, b in enumerate(final_buildings):
            poly, bldg_shape, windows, dividers, b_floors, b_units, b_ns_dist, name = b
            x, y = poly.exterior.xy
            color = get_type_color(name)
            
            hover_text = f"동: {idx+1}동<br>타입: {name}<br>층수: {b_floors}F<br>층당 세대수: {b_units}세대 (동 전체 {b_units * b_floors}세대)<br>인동간격: {b_ns_dist:.1f}m"
            fig.add_trace(go.Scatter(x=list(x), y=list(y), fill='toself', fillcolor=color, mode='lines', line=dict(color='#333333', width=1), text=hover_text, hoverinfo='text', showlegend=False))
            
            div_x, div_y = [], []
            for ((x1, y1), (x2, y2)) in dividers:
                div_x.extend([x1, x2, None])
                div_y.extend([y1, y2, None])
            if div_x:
                fig.add_trace(go.Scatter(x=div_x, y=div_y, mode='lines', line=dict(color='white', width=1.5), hoverinfo='skip', showlegend=False))
            
            for (wx1, wy1, wx2, wy2, dx, dy) in windows:
                proj_poly = Polygon([(wx1, wy1), (wx2, wy2), (wx2 + dx * b_ns_dist, wy2 + dy * b_ns_dist), (wx1 + dx * b_ns_dist, wy1 + dy * b_ns_dist)])
                px, py = proj_poly.exterior.xy
                fig.add_trace(go.Scatter(x=list(px), y=list(py), fill='toself', fillcolor='rgba(255,0,0,0.1)', mode='lines', line=dict(color='rgba(255,0,0,0.7)', width=1, dash='dot'), hoverinfo='skip', showlegend=False))





        b_minx, b_miny, b_maxx, b_maxy = base_site_poly.bounds
        plot_min_x, plot_max_x = b_minx - 15, b_maxx + 15
        plot_min_y, plot_max_y = b_miny - 15, b_maxy + 15

        fig.update_layout(
            xaxis=dict(scaleanchor="y", scaleratio=1, visible=True, range=[plot_min_x, plot_max_x], title="가로 (m)"),
            yaxis=dict(visible=True, range=[plot_min_y, plot_max_y], title="세로 (m)"),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, title="평형 (전용면적)", bgcolor='rgba(255,255,255,0.8)', bordercolor='black', borderwidth=1),
            hovermode='closest',
            height=850
        )

        st.plotly_chart(fig, use_container_width=True)

# Trigger reload
