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
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

UNIT_TYPES = {
    "26㎡(복도식)": (20.0, 12.0, 4, 26.0, "판상형"),
    "31㎡(복도식)": (24.0, 12.0, 4, 31.0, "판상형"),
    "36㎡(복도식)": (28.0, 12.0, 4, 36.0, "판상형"),
    "41㎡(복도식)": (32.0, 12.0, 4, 41.0, "판상형"),
    "46㎡(복도식)": (36.0, 12.0, 4, 46.0, "판상형"),
    "55㎡(계단식)": (22.0, 14.0, 2, 55.0, "판상형"),
    "59㎡(계단식)": (24.0, 14.0, 2, 59.0, "판상형"),
    "65㎡(계단식)": (26.0, 15.0, 2, 65.0, "판상형"),
    "74㎡(계단식)": (28.0, 15.5, 2, 74.0, "판상형"),
    "84㎡(계단식)": (30.0, 16.0, 2, 84.0, "판상형"),
    "84㎡(탑상형)": (24.0, 24.0, 3, 84.0, "L자형")
}

SIZE_COLORS = {
    "84㎡(탑상형)": "#3cb371",
    "84㎡": "#6495ed",
    "59㎡": "#00ced1",
    "74㎡": "#ff7f50",
    "65㎡": "#dda0dd",
    "55㎡": "#ffd700",
    "46㎡": "#4682b4",
    "41㎡": "#ff69b4",
    "36㎡": "#cd5c5c",
    "31㎡": "#8fbc8f",
    "26㎡": "#9370db"
}

def get_base_windows(b_w, b_l, bldg_shape, units):
    windows = []
    if bldg_shape == "판상형":
        step = b_w / units
        for i in range(units):
            windows.append((i * step, 0, (i + 1) * step, 0, 0, -1))
    elif bldg_shape in ["타워형", "L자형"]:
        t = b_w / 3.0
        windows.append((t, 0, 2*t, 0, 0, -1))
        windows.append((2*t, 0, 3*t, 0, 0, -1))
        windows.append((0, 0, 0, t, -1, 0))
        windows.append((0, t, 0, 2*t, -1, 0))
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
        dividers.append(((t*2, 0), (t*2, t)))
        dividers.append(((0, t), (t, t)))
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

@st.cache_data(show_spinner=False)
def auto_optimize_layout(site_w, site_l, site_shape_type, floors, h_multiplier, setback_x, setback_y, min_ns_dist, side_dist, max_far, max_bcr, selected_sizes, size_ratios, school_n, school_s, school_e, school_w, road_n, road_s, road_e, road_w, sunlight_dir, trap_bottom=0, trap_top=0, trap_height=0, l_w=0, l_l=0, l_w_inner=0, l_l_inner=0, exclude_balcony=False, layout_version=18):
    
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
        
    site_area = base_site_poly.area
    # 다각형의 안쪽으로 setback만큼 쪼그라든(Shrink) 실제 건축가능 영역 생성
    site_poly = base_site_poly.buffer(-setback_x)
    
    # 바운딩 박스 기준으로 학교 및 북쪽 기준선 설정
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
    
    selected_keys = [k for k in UNIT_TYPES.keys() if any(size in k for size in selected_sizes)]
    for name in selected_keys:
        size_label = name.split("(")[0]
        if size_ratios.get(size_label, 0) <= 0:
            continue
        w, l, units, area, shape = UNIT_TYPES[name]
        poly = create_building_poly(w, l, shape)
        windows = get_base_windows(w, l, shape, units)
        dividers = get_base_dividers(w, l, shape, units)
        types_info.append({
            'name': name, 'size_label': size_label, 'poly': poly, 'units': units, 'shape': shape,
            'area': poly.area, 'windows': windows, 'dividers': dividers
        })
    
    points = []
    # 대지가 커지면 연산 속도를 위해 탐색 간격을 넓힘
    step_size = max(15.0, min(site_w, site_l) / 30.0)
    for x in np.arange(setback_x, site_w - setback_x, step_size):
        for y in np.arange(setback_y, site_l - setback_y, step_size):
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
                
                far_multiplier = (0.70 if t_info['shape'] == '판상형' else 0.85) if exclude_balcony else 1.0
                new_far = ((total_floor_area + (t_info['area'] * far_multiplier) * b_floors) / site_area) * 100
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
                        minx, miny, maxx, maxy = candidate.bounds
                        for t_hr in [9.0, 12.0, 15.0]:
                            if conflict: break
                            az_deg = 180 + (t_hr - 12) * 15
                            alt_deg = 29.0 - abs(t_hr - 12) * 4.0
                            s_len = b_h / math.tan(math.radians(alt_deg))
                            sdx = math.sin(math.radians(az_deg - 180)) * s_len
                            sdy = math.cos(math.radians(az_deg - 180)) * s_len
                            
                            # Fast AABB check
                            s_minx = min(minx, minx + sdx)
                            s_maxx = max(maxx, maxx + sdx)
                            s_miny = min(miny, miny + sdy)
                            s_maxy = max(maxy, maxy + sdy)
                            
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
                                
                    # 0-1. 일조권 사선제한 (H/2)
                    if not conflict:
                        bldg_min_x, bldg_min_y, bldg_max_x, bldg_max_y = candidate.bounds
                        if sunlight_dir == "정북방향":
                            if (maxy - bldg_max_y) < (b_h / 2.0) - road_n:
                                conflict = True
                        else:  # 정남방향
                            if (bldg_min_y - miny) < (b_h / 2.0) - road_s:
                                conflict = True
                            
                    # 0-2. 채광창 방향 대지경계선 이격 (0.5H) (도로 너비만큼 완화)
                    if not conflict:
                        window_setback = b_h * 0.5
                        for (wx1, wy1, wx2, wy2, dx, dy) in final_windows:
                            min_wy, max_wy = min(wy1, wy2), max(wy1, wy2)
                            min_wx, max_wx = min(wx1, wx2), max(wx1, wx2)
                            if dy > 0.1:  # 북향
                                if (maxy - max_wy) < window_setback - road_n: conflict = True
                            if dy < -0.1: # 남향
                                if (min_wy - miny) < window_setback - road_s: conflict = True
                            if dx > 0.1:  # 동향
                                if (maxx - max_wx) < window_setback - road_e: conflict = True
                            if dx < -0.1: # 서향
                                if (min_wx - minx) < window_setback - road_w: conflict = True
                            if conflict: break

                    if conflict: continue
                    
                    c_minx, c_miny, c_maxx, c_maxy = candidate.bounds
                    for existing_bldg, _, ex_windows, _, ex_floors, _, ex_ns_dist, _ in buildings:
                        e_minx, e_miny, e_maxx, e_maxy = existing_bldg.bounds
                        max_possible_dist = max(b_ns_dist, ex_ns_dist) + 5.0
                        if c_maxx < e_minx - max_possible_dist or c_minx > e_maxx + max_possible_dist or c_maxy < e_miny - max_possible_dist or c_miny > e_maxy + max_possible_dist:
                            continue
                        if candidate.intersects(existing_bldg.buffer(side_dist, resolution=2)):
                            conflict = True
                            break
                        
                        for (wx1, wy1, wx2, wy2, dx, dy) in final_windows:
                            proj_poly = Polygon([(wx1, wy1), (wx2, wy2), (wx2 + dx * b_ns_dist, wy2 + dy * b_ns_dist), (wx1 + dx * b_ns_dist, wy1 + dy * b_ns_dist)])
                            if proj_poly.intersects(existing_bldg):
                                conflict = True
                                break
                        if conflict: break
                        
                        for (wx1, wy1, wx2, wy2, dx, dy) in ex_windows:
                            proj_poly = Polygon([(wx1, wy1), (wx2, wy2), (wx2 + dx * ex_ns_dist, wy2 + dy * ex_ns_dist), (wx1 + dx * ex_ns_dist, wy1 + dy * ex_ns_dist)])
                            if proj_poly.intersects(candidate):
                                conflict = True
                                break
                        if conflict: break
                            
                    if not conflict:
                        final_dividers = transform_dividers(t_info['dividers'], angle, x, y)
                        buildings.append((candidate, t_info['shape'], final_windows, final_dividers, b_floors, t_info['units'], b_ns_dist, t_info['name']))
                        total_floor_area += (t_info['area'] * far_multiplier) * b_floors
                        bldg_area += t_info['area']
                        added_units = t_info['units'] * b_floors
                        total_units += added_units
                        placed_counts[t_info['size_label']] += added_units
                        placed = True
                        break
                    
    return buildings, site_area, bldg_area

st.set_page_config(layout="wide", page_title="속 터져서 내가 직접 만들어본 공동주택 가배치")
st.title("속 터져서 내가 직접 만들어 본 공동주택 假배치 😤")
st.markdown("<div style='background-color: #ffe066; padding: 5px 15px; border-radius: 5px; display: inline-block; font-weight: 800; font-size: 1.1em; color: #333333; margin-bottom: 20px;'>© 2026 김진우 (Jinwoo Kim). All rights reserved.</div>", unsafe_allow_html=True)

col_input, col_viz = st.columns([1, 2])

with col_input:
    st.header("1. 일조 시뮬레이션 (동지 기준)")
    time_of_day = st.slider("시간대 (Time of Day)", min_value=9.0, max_value=15.0, value=12.0, step=0.5, format="%.1f 시")
    
    st.header("2. 대지 및 법규 조건")
    site_shape_type = st.selectbox("대지 형상 (Site Shape)", ["직사각형", "L자형", "ㄱ자형", "사다리꼴"])
    
    # 기본값 초기화
    trap_bottom, trap_top, trap_height = 0, 0, 0
    l_w, l_l, l_w_inner, l_l_inner = 0, 0, 0, 0
    
    if site_shape_type == "직사각형":
        st.caption("※ 선택한 형상에 맞게 가로/세로 길이로 다각형 대지가 생성됩니다.")
        site_w = st.number_input("대지 가로 길이 (m)", min_value=30, max_value=2000, value=200, step=10)
        site_l = st.number_input("대지 세로 길이 (m)", min_value=30, max_value=2000, value=200, step=10)
    elif site_shape_type == "사다리꼴":
        st.caption("※ 사다리꼴의 치수를 상세 입력합니다.")
        trap_bottom = st.number_input("아랫변 길이 (긴변, m)", min_value=30, max_value=2000, value=250, step=10)
        trap_top = st.number_input("윗변 길이 (짧은변, m)", min_value=10, max_value=2000, value=150, step=10)
        trap_height = st.number_input("높이 (m)", min_value=30, max_value=2000, value=200, step=10)
        site_w = max(trap_bottom, trap_top)
        site_l = trap_height
    elif site_shape_type == "L자형" or site_shape_type == "ㄱ자형":
        st.caption("※ 선택하신 대지의 치수를 상세 입력합니다.")
        col_w1, col_w2 = st.columns(2)
        l_w = col_w1.number_input("가로 전체 길이 (m)", min_value=30, max_value=2000, value=250, step=10)
        l_w_inner = col_w2.number_input("파인 부분 가로 (m)", min_value=10, max_value=400, value=120, step=10)
        col_l1, col_l2 = st.columns(2)
        l_l = col_l1.number_input("세로 전체 길이 (m)", min_value=30, max_value=2000, value=200, step=10)
        l_l_inner = col_l2.number_input("파인 부분 세로 (m)", min_value=10, max_value=400, value=100, step=10)
        # 예외 처리
        l_w_inner = min(l_w_inner, l_w - 10)
        l_l_inner = min(l_l_inner, l_l - 10)
        site_w = l_w
        site_l = l_l
    max_far = st.number_input("용적률 상한 (%)", min_value=50, max_value=1000, value=300, step=10)
    max_bcr = st.number_input("건폐율 상한 (%)", min_value=10, max_value=100, value=20, step=2)
    exclude_balcony = st.checkbox("서비스면적(발코니) 용적률 제외 보정", value=True, help="실제 아파트처럼 발코니 면적을 용적률 산정에서 제외하여 세대수를 극대화합니다.")
    limit_floors = st.checkbox("층수 제한 있음", value=True)
    if limit_floors:
        floors = st.number_input("최고 층수 제한 (층)", min_value=1, max_value=100, value=35, step=1)
    else:
        floors = 50
        
    st.subheader("일조권 사선제한 방향")
    sunlight_dir = st.radio("적용 기준 (건축법 제61조)", ["정북방향", "정남방향"], horizontal=True, index=1,
                            help="일반적으로 정북방향이 적용되나, 택지개발지구/지구단위계획 등에서는 정남방향이 적용될 수 있습니다.")
        
    st.subheader("교육환경보호구역 (학교 연접 여부)")
    st.caption("※ 학교 사이에 도로/녹지가 있다면 아래 '인접대지' 체크를 꼭 해제해주세요!")
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
    
    st.header("3. 평형 선택 (전용면적 기준)")
    st.caption("배치에 사용할 평형을 모두 선택하세요 (판상형/타워형은 자동 혼합 최적화)")
    
    st.write("초소형(복도식)")
    c1, c2, c3, c4, c5 = st.columns(5)
    use_26 = c1.checkbox("26㎡", value=False)
    use_31 = c2.checkbox("31㎡", value=False)
    use_36 = c3.checkbox("36㎡", value=False)
    use_41 = c4.checkbox("41㎡", value=False)
    use_46 = c5.checkbox("46㎡", value=False)
    st.write("소형/중형(계단식)")
    c6, c7, c8, c9, c10 = st.columns(5)
    use_55 = c6.checkbox("55㎡", value=False)
    use_59 = c7.checkbox("59㎡", value=False)
    use_65 = c8.checkbox("65㎡", value=False)
    use_74 = c9.checkbox("74㎡", value=False)
    use_84 = c10.checkbox("84㎡", value=True)
    selected_sizes = []
    if use_26: selected_sizes.append("26㎡")
    if use_31: selected_sizes.append("31㎡")
    if use_36: selected_sizes.append("36㎡")
    if use_41: selected_sizes.append("41㎡")
    if use_46: selected_sizes.append("46㎡")
    if use_55: selected_sizes.append("55㎡")
    if use_59: selected_sizes.append("59㎡")
    if use_65: selected_sizes.append("65㎡")
    if use_74: selected_sizes.append("74㎡")
    if use_84: selected_sizes.append("84㎡")
    
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
    
    st.header("4. 법정 제약조건 설정")
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
    - **정북방향 사선제한:** 북쪽 경계로부터 H/2 이상 이격 (건축법)
    - **채광창 대지경계 이격:** 창문 방향 인접대지로부터 0.5H 이상 이격
    """)

with col_viz:
    st.info("💡 **교육환경보호 (일조권):** 학교 부지가 설정되면, 오전 9시~오후 3시 사이에 학교로 그림자를 드리우는 건물을 알아서 층수를 깎아(Step-down) 배치합니다.\n\n💡 **대지경계 사선제한:** 주변에 빈 땅(도로)이 있을 경우 그 너비만큼 일조권(H/2) 및 이격(0.5H) 제한을 완화받아 자동 배치됩니다.\n\n⚠️ **참고:** 도면에 그려지는 붉은색 투영 면적은 '건물 간 인동간격(0.8H)' 확인용입니다. 이 면적이 대지경계선 밖(도로 등)으로 튀어나가는 것은 합법입니다.")
    inputs_tuple = (
        time_of_day, site_shape_type, trap_bottom, trap_top, trap_height, 
        l_w, l_l, l_w_inner, l_l_inner, max_far, max_bcr, exclude_balcony, 
        limit_floors, floors, sunlight_dir, school_n, school_s, school_e, school_w, 
        road_n, road_s, road_e, road_w, str(selected_sizes), str(size_ratios)
    )

    calc_btn = st.button("🚀 시뮬레이션 계산 시작", type="primary", use_container_width=True)
    
    if calc_btn:
        st.session_state['last_inputs'] = inputs_tuple
        with st.spinner("AI가 최적의 배치를 찾고 있습니다... (약 10~20초 소요)"):
            bldgs, s_area, b_area = auto_optimize_layout(
                site_w, site_l, site_shape_type, floors, h_multiplier, setback_x, setback_y, min_ns_dist, side_dist, max_far, max_bcr, selected_sizes, size_ratios, school_n, school_s, school_e, school_w, road_n, road_s, road_e, road_w, sunlight_dir, trap_bottom=trap_bottom, trap_top=trap_top, trap_height=trap_height, l_w=l_w, l_l=l_l, l_w_inner=l_w_inner, l_l_inner=l_l_inner, exclude_balcony=exclude_balcony, layout_version=18)
            st.session_state['sim_result'] = (bldgs, s_area, b_area)

    if 'last_inputs' in st.session_state and st.session_state['last_inputs'] != inputs_tuple:
        st.warning("⚠️ 입력값이 변경되었습니다. 적용하려면 '🚀 시뮬레이션 계산 시작' 버튼을 다시 눌러주세요.")
        st.stop()
        
    if 'sim_result' not in st.session_state:
        st.info("💡 대지 조건과 법규를 모두 입력하신 후, 위의 '🚀 시뮬레이션 계산 시작' 버튼을 눌러주세요!")
        st.stop()
        
    buildings, site_area, bldg_area = st.session_state['sim_result']
        
    main_viz = st.container()
    override_viz = st.container()
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
    
    for i, b in enumerate(buildings):
        poly, shape, windows, dividers, b_floors, units, ns_dist, name = b
        row = edited_df.iloc[i]
        
        # Override logic
        if row["수동 고정"]:
            b_floors = row["강제 층수"]
            
        if b_floors == 0:
            continue
            
        final_buildings.append((poly, shape, windows, dividers, b_floors, units, ns_dist, name))
        
        far_multiplier = (0.70 if shape == '판상형' else 0.85) if exclude_balcony else 1.0
        total_floor_area += (poly.area * far_multiplier) * b_floors
        total_units += units * b_floors
        actual_bldg_area += poly.area
        
    bcr = (actual_bldg_area / site_area) * 100 if site_area > 0 else 0
    far = (total_floor_area / site_area) * 100 if site_area > 0 else 0
    
    with main_viz:
        st.header("배치 및 그림자 간섭 분석")
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        metrics_col1.metric("최대 달성 세대수", f"{total_units:,} 세대")
        metrics_col2.metric("세팅된 동 수", f"{len(final_buildings)} 동")
        metrics_col3.metric("건폐율 (BCR)", f"{bcr:.2f} %")
        metrics_col4.metric("용적률 (FAR)", f"{far:.2f} %")
        
        azimuth_deg = 180 + (time_of_day - 12) * 15
        altitude_deg = 29.0 - abs(time_of_day - 12) * 4.0
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
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
            
        site_poly = base_site_poly.buffer(-setback_x)

        bx, by = base_site_poly.exterior.xy
        ax.plot(bx, by, color='black', linewidth=2)
        ax.fill(bx, by, color='#f0f8ff')
        
        ix, iy = site_poly.exterior.xy
        ax.plot(ix, iy, color='red', linewidth=1, linestyle='--')
        
        minx, miny, maxx, maxy = base_site_poly.bounds
        
        # 학교 영역 그리기 (도로 너비 반영)
        off_n = max(5.0, road_n)
        if school_n:
            ax.add_patch(patches.Rectangle((minx, maxy + off_n), maxx - minx, 100, linewidth=2, edgecolor='#e6b800', facecolor='#fffacd', hatch='//'))
            ax.text((minx+maxx)/2, maxy + off_n + 50, "북쪽 학교", color='#8b6508', ha='center', va='center', fontsize=14, fontweight='bold')
        off_s = max(5.0, road_s)
        if school_s:
            ax.add_patch(patches.Rectangle((minx, miny - off_s - 100), maxx - minx, 100, linewidth=2, edgecolor='#e6b800', facecolor='#fffacd', hatch='//'))
            ax.text((minx+maxx)/2, miny - off_s - 50, "남쪽 학교", color='#8b6508', ha='center', va='center', fontsize=14, fontweight='bold')
        off_e = max(5.0, road_e)
        if school_e:
            ax.add_patch(patches.Rectangle((maxx + off_e, miny), 100, maxy - miny, linewidth=2, edgecolor='#e6b800', facecolor='#fffacd', hatch='//'))
            ax.text(maxx + off_e + 50, (miny+maxy)/2, "동쪽 학교", color='#8b6508', ha='center', va='center', fontsize=14, fontweight='bold', rotation=270)
        off_w = max(5.0, road_w)
        if school_w:
            ax.add_patch(patches.Rectangle((minx - off_w - 100, miny), 100, maxy - miny, linewidth=2, edgecolor='#e6b800', facecolor='#fffacd', hatch='//'))
            ax.text(minx - off_w - 50, (miny+maxy)/2, "서쪽 학교", color='#8b6508', ha='center', va='center', fontsize=14, fontweight='bold', rotation=90)

        
        min_extent_x, max_extent_x = 0, site_w
        min_extent_y, max_extent_y = 0, site_l
        
        for idx, (poly, bldg_shape, windows, dividers, b_floors, b_units, b_ns_dist, name) in enumerate(final_buildings):
            b_h = b_floors * 3.0
            shadow_length = b_h / math.tan(math.radians(altitude_deg))
            shadow_dx = math.sin(math.radians(azimuth_deg - 180)) * shadow_length
            shadow_dy = math.cos(math.radians(azimuth_deg - 180)) * shadow_length
            
            shifted_poly = translate(poly, xoff=shadow_dx, yoff=shadow_dy)
            shadow_poly = unary_union([poly, shifted_poly]).convex_hull
            
            sx, sy = shadow_poly.exterior.xy
            ax.fill(sx, sy, alpha=0.3, facecolor='black', edgecolor='none')
            
            # 그림자 범위도 포함
            min_extent_x = min(min_extent_x, min(sx))
            max_extent_x = max(max_extent_x, max(sx))
            min_extent_y = min(min_extent_y, min(sy))
            max_extent_y = max(max_extent_y, max(sy))
            
        for idx, (poly, bldg_shape, windows, dividers, b_floors, b_units, b_ns_dist, name) in enumerate(final_buildings):
            x, y = poly.exterior.xy
            
            
            color = SIZE_COLORS.get(name, SIZE_COLORS.get(name.split('(')[0], '#cccccc'))
            
            ax.fill(x, y, alpha=0.9, facecolor=color, edgecolor='#333333', linewidth=1)
            
            for ((x1, y1), (x2, y2)) in dividers:
                ax.plot([x1, x2], [y1, y2], color='white', linewidth=1.5, linestyle='-')
            
            cx, cy = poly.centroid.coords[0]
            label_text = f"{idx+1}동\n{b_floors}F x {b_units}호"
            ax.text(cx, cy, label_text, color='white', ha='center', va='center', fontsize=9, fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.3, edgecolor='none', boxstyle='round,pad=0.2'))
            
            for (wx1, wy1, wx2, wy2, dx, dy) in windows:
                proj_poly = Polygon([(wx1, wy1), (wx2, wy2), (wx2 + dx * b_ns_dist, wy2 + dy * b_ns_dist), (wx1 + dx * b_ns_dist, wy1 + dy * b_ns_dist)])
                px, py = proj_poly.exterior.xy
                ax.plot(px, py, color='red', linewidth=1, linestyle=':', alpha=0.7)
                ax.fill(px, py, color='red', alpha=0.1)
                
                ex = (wx1 + wx2) / 2 + dx * b_ns_dist
                ey = (wy1 + wy2) / 2 + dy * b_ns_dist
                
                ax.text(ex, ey, f" {b_ns_dist:.1f}m", color='darkred', fontsize=8, fontweight='bold',
                        ha='center' if abs(dx) < 0.1 else ('left' if dx > 0 else 'right'),
                        va='center' if abs(dy) < 0.1 else ('bottom' if dy > 0 else 'top'))
                
                # 투영면 범위 포함
                min_extent_x = min(min_extent_x, min(px))
                max_extent_x = max(max_extent_x, max(px))
                min_extent_y = min(min_extent_y, min(py))
                max_extent_y = max(max_extent_y, max(py))

        ax.annotate('N', xy=(site_w - 20, site_l - 10), xytext=(site_w - 20, site_l - 30),
                    arrowprops=dict(facecolor='black', shrink=0, width=3, headwidth=10),
                    fontsize=16, fontweight='bold', ha='center', va='top')
                    
        import matplotlib.patches as mpatches
        legend_patches = []
        # Create a set of sizes actually placed
        placed_sizes = set([b[7] for b in final_buildings])
        for size in sorted(list(placed_sizes)):
            legend_patches.append(mpatches.Patch(color=SIZE_COLORS.get(size, SIZE_COLORS.get(size.split('(')[0], '#cccccc')), label=size.replace('계단식', '').replace('()', '')))
        if legend_patches:
            ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.15, 1.0), title="평형 (전용면적)", title_fontsize='10', fontsize='9')

        # 계산된 범위를 바탕으로 여유를 두고 화면 설정
        ax.set_xlim(min(-10, min_extent_x - 15), max(site_w + 10, max_extent_x + 15))
        ax.set_ylim(min(-10, min_extent_y - 15), max(site_l + 10, max_extent_y + 15))
        ax.set_aspect('equal')
        ax.set_xlabel("Width (m)")
        ax.set_ylabel("Length (m)")
        
        st.pyplot(fig)
