import csv
import time
import math
from utils import *
from shapely.geometry import LineString, Polygon  # 注意，自己写的包，要放到调用的包的前面
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory

plt.figure()
# variable definitions
boundary_path = r"demo_boundary.csv"
init_ref_path = r"refernce_right.csv"
work_width = 6  # working width
turn_radius = 7  # minimum turnning radius
k_max = 0.120  # 1 / turn_radius  maximum steering curvature
ext_length = 10  # extended length of the reference side in boundary
pt_num = 30  # the number of characteristic points of the generated cubic B-spline curve
pic_num = 1000  # the number of characteristic points used to display the cubic B-spline curve
p_dir = 'right'  # propagation direction
gama = 0.8  # γ, adjustable value, the allowable deviation from the working width
mu = 0.01  # μ, adjustable value, the allowable deviation of offset direction from the normal direction
straightness_cost_weight = 1  # W1
length_cost_weight = 1  # W2
swath_width_cost_weight = 1  # W3

# 1.obtaining field boundary from csv
boundary_xy = []
with open(boundary_path, 'r', encoding='GBK') as fp:
    reader = csv.DictReader(fp)
    for x in reader:
        temp = [float(x['x']), float(x['y'])]
        boundary_xy.append(temp)
boundary = LineString(boundary_xy)
boundary_x = [i[0] for i in boundary_xy]
boundary_y = [i[1] for i in boundary_xy]
plt.plot(boundary_x, boundary_y, 'k')

# 2.generating the initial reference line
# 1)generating the headland passes and inner boundary
alpha = 1
headland_path_num = math.ceil(turn_radius * alpha / work_width)
headland_path = [boundary.parallel_offset(work_width / 2, 'right')]
for i in range(headland_path_num - 1):
    headland_path.append(headland_path[-1].parallel_offset(work_width, 'left'))
inner_boundary = headland_path[-1].parallel_offset(work_width / 2, 'left')
inner_boundary_xy = list(inner_boundary.coords)
inner_boundary_x = [i[0] for i in inner_boundary_xy]
inner_boundary_y = [i[1] for i in inner_boundary_xy]
plt.plot(inner_boundary_x, inner_boundary_y, 'k')

# csv_matrix = []
# csv_table_headers = ['x', 'y']
# csv_matrix.append(csv_table_headers)
# for i in range(len(inner_boundary_x)):
#     csv_row = [inner_boundary_x[i], inner_boundary_y[i]]
#     csv_matrix.append(csv_row)
# for i in range(len(csv_matrix)):
#     f = open(r"inner_boundary.csv", 'a', newline='')
#     writer = csv.writer(f)
#     writer.writerow(csv_matrix[i])
#     f.close()

# 2）obtaining reference line from csv
init_ref_xy = []
with open(init_ref_path, 'r', encoding='GBK') as fp:
    reader = csv.DictReader(fp)
    for x in reader:
        temp = [float(x['x']), float(x['y'])]
        init_ref_xy.append(temp)
init_ref = LineString(init_ref_xy)
init_ref_x = [i[0] for i in init_ref_xy]
init_ref_y = [i[1] for i in init_ref_xy]
plt.plot(init_ref_x, init_ref_y, 'r')
# 3）extending both ends of the selected side
init_ref_ext_x, init_ref_ext_y = line_extension(init_ref_x, init_ref_y, ext_length)
# 4）fiting the initial reference curve using cubic B-spline
init_ref_spl_x, init_ref_spl_y, init_u, init_tck = generate_cubic_Bspline_approxi(init_ref_ext_x, init_ref_ext_y,
                                                                                  pt_num)
cur = calc_curvature(init_u, init_tck)
print(cur)

# variables used in quadratic programming
path_x_list = []
path_y_list = []
path_x_clipped = []
path_y_clipped = []
path_x_pro1 = []
path_y_pro1 = []
path_x_pro2 = []
path_y_pro2 = []
next_u = []
next_tck = []
wayline_cur = []
offset_angle_diff = []
overlaps = []
skips = []

for id in range(100):
    # 3.quadratic programming
    # 1）calculating the parameters used in objective function and constraints
    start = time.time()
    if id == 0:  # first smoothing the initial reference line
        Ng = len(init_ref_spl_x)  # the number of the waypoint
        ref_x = init_ref_spl_x
        ref_y = init_ref_spl_y
        d_s = calc_average_distance(ref_x, ref_y)
        # the unit normal vectors of the points on the initial reference line
        ref_e_x, ref_e_y = calc_unit_normal_vector(init_u, init_tck, direction=p_dir)
        print(ref_e_x)
        print(ref_e_y)
        offset_dis = 0.5
    elif id == 1:  # the reference line of the first wayline is the initial reference line, and the offset distance is work_width/2
        path_x_pro1, path_y_pro1, next_u, next_tck = generate_cubic_Bspline(path_x_pro2,
                                                                            path_y_pro2,
                                                                            pt_num)
        Ng = len(path_x_pro1)
        ref_x = path_x_pro1
        ref_y = path_y_pro1
        d_s = calc_average_distance(ref_x, ref_y)
        ref_e_x, ref_e_y = calc_unit_normal_vector(next_u, next_tck, direction=p_dir)
        offset_dis = work_width / 2
    else:  # work_width the reference line of the latter wayline is the last wayline, and the offset distance is work_width
        # the first treatment to avoid self-intersection: resampling waypoints on the reference line
        path_x_pro1, path_y_pro1, next_u, next_tck = generate_cubic_Bspline(path_x_pro2,
                                                                            path_y_pro2,
                                                                            pt_num)
        Ng = len(path_x_pro1)
        ref_x = path_x_pro1
        ref_y = path_y_pro1
        d_s = calc_average_distance(ref_x, ref_y)
        ref_e_x, ref_e_y = calc_unit_normal_vector(next_u, next_tck, direction=p_dir)
        offset_dis = work_width

    # end_x = [ref_x[i] + 6 * ref_e_x[i] for i in range(30)]
    # end_y = [ref_y[i] + 6 * ref_e_y[i] for i in range(30)]
    # for i in range(30):
    #     plt.plot([ref_x, end_x], [ref_y, end_y], '-')

    # 2）initialising the model
    model = pyo.ConcreteModel()
    # 3）decision variables
    model.x = pyo.Var(range(Ng))
    model.y = pyo.Var(range(Ng))
    model.s = pyo.Var(range(1, Ng - 1), bounds=(1e-20, float('inf')))
    x = model.x
    y = model.y
    s = model.s
    # 4）constraints
    # s.t.1,s.t.2——swath width constraints
    model.swath_width_x_lower = pyo.ConstraintList()
    model.swath_width_x_upper = pyo.ConstraintList()
    model.swath_width_y_lower = pyo.ConstraintList()
    model.swath_width_y_upper = pyo.ConstraintList()
    for i in range(Ng):
        # model.swath_width_x_lower.add(expr=(x[i] - ref_x[i]) / ref_e_x[i] >= (1 - gama_l) * offset_dis)
        # model.swath_width_x_upper.add(expr=(x[i] - ref_x[i]) / ref_e_x[i] <= (1 + gama_u) * offset_dis)
        # model.swath_width_y_lower.add(expr=(y[i] - ref_y[i]) / ref_e_y[i] >= (1 - gama_l) * offset_dis)
        # model.swath_width_y_upper.add(expr=(y[i] - ref_y[i]) / ref_e_y[i] <= (1 + gama_u) * offset_dis)
        lx = ref_x[i] + min(((1 - gama) * offset_dis) * ref_e_x[i], ((1 + gama) * offset_dis) * ref_e_x[i])
        ux = ref_x[i] + max(((1 - gama) * offset_dis) * ref_e_x[i], ((1 + gama) * offset_dis) * ref_e_x[i])

        ly = ref_y[i] + min(((1 - gama) * offset_dis) * ref_e_y[i], ((1 + gama) * offset_dis) * ref_e_y[i])
        uy = ref_y[i] + max(((1 - gama) * offset_dis) * ref_e_y[i], ((1 + gama) * offset_dis) * ref_e_y[i])
        model.swath_width_x_lower.add(expr=x[i] >= lx)
        model.swath_width_x_upper.add(expr=x[i] <= ux)
        model.swath_width_y_lower.add(expr=y[i] >= ly)
        model.swath_width_y_upper.add(expr=y[i] <= uy)
    # s.t.3——offset direction constraints
    model.offset_direction_lower = pyo.ConstraintList()
    model.offset_direction_upper = pyo.ConstraintList()
    for i in range(Ng):
        model.offset_direction_lower.add(expr=(y[i] - ref_y[i]) / (x[i] - ref_x[i]) >= (ref_e_y[i] / ref_e_x[i]) - mu)
        model.offset_direction_upper.add(expr=(y[i] - ref_y[i]) / (x[i] - ref_x[i]) <= (ref_e_y[i] / ref_e_x[i]) + mu)
    # s.t.4——curvature constraints
    model.curvature = pyo.ConstraintList()
    for i in range(1, Ng - 1):
        F_ref = (ref_x[i - 1] + ref_x[i + 1] - 2 * ref_x[i]) ** 2 + (ref_y[i - 1] + ref_y[i + 1] - 2 * ref_y[i]) ** 2
        F_fdp_x_f = 2 * (ref_x[i - 1] + ref_x[i + 1] - 2 * ref_x[i])
        F_fdp_y_f = 2 * (ref_y[i - 1] + ref_y[i + 1] - 2 * ref_y[i])
        F_fdp_x_m = -4 * (ref_x[i - 1] + ref_x[i + 1] - 2 * ref_x[i])
        F_fdp_y_m = -4 * (ref_y[i - 1] + ref_y[i + 1] - 2 * ref_y[i])
        F_fdp_x_l = 2 * (ref_x[i - 1] + ref_x[i + 1] - 2 * ref_x[i])
        F_fdp_y_l = 2 * (ref_y[i - 1] + ref_y[i + 1] - 2 * ref_y[i])
        F_fdp_X_ref_multi_X_ref = F_fdp_x_f * ref_x[i - 1] + F_fdp_y_f * ref_y[i - 1] + \
                                  F_fdp_x_m * ref_x[i] + F_fdp_y_m * ref_y[i] + \
                                  F_fdp_x_l * ref_x[i + 1] + F_fdp_y_l * ref_y[i + 1]
        expr_variable = F_fdp_x_f * x[i - 1] + F_fdp_y_f * y[i - 1] + \
                        F_fdp_x_m * x[i] + F_fdp_y_m * y[i] + \
                        F_fdp_x_l * x[i + 1] + F_fdp_y_l * y[i + 1] - s[i]
        expr_constant = (d_s ** 2 * k_max) ** 2 - F_ref + F_fdp_X_ref_multi_X_ref
        model.curvature.add(expr=expr_variable <= expr_constant)
    # 5）objective function
    straightness_cost = 0
    length_cost = 0
    swath_width_cost = 0
    for i in range(Ng):
        x_expectant = ref_x[i] + ref_e_x[i] * offset_dis
        y_expectant = ref_y[i] + ref_e_y[i] * offset_dis
        swath_width_cost += (x[i] - x_expectant) ** 2 + (y[i] - y_expectant) ** 2
        if i != 0 and i != Ng - 1:
            straightness_cost += (x[i - 1] + x[i + 1] - 2 * x[i]) ** 2 + (y[i - 1] + y[i + 1] - 2 * y[i]) ** 2
        if i != Ng - 1:
            length_cost += (x[i + 1] - x[i]) ** 2 + (y[i + 1] - y[i]) ** 2
    model.obj = pyo.Objective(
        expr=straightness_cost_weight * straightness_cost + length_cost_weight * length_cost + swath_width_cost_weight * swath_width_cost,
        sense=minimize)
    # 6）set the solver and solving the quadratic programming problem
    opt = SolverFactory('ipopt')
    # opt = SolverFactory('gurobi')
    opt.options['max_iter'] = 5000
    result = opt.solve(model)
    end = time.time()
    print('time:', end - start, 's')
    # 7）obtaining the solutions
    path_x = [pyo.value(x[i]) for i in range(Ng)]
    path_y = [pyo.value(y[i]) for i in range(Ng)]

    # calculating the curvature of the waypoints on new generated wayline
    tck, u_origin = interpolate.splprep([path_x, path_y], k=3, s=0)
    print(calc_curvature(u_origin, tck))

    # calculating the actual swath width, overlap and skip of the waypoints
    width = []
    overlap = []
    skip = []
    for i in range(Ng):
        width.append((path_x[i] - ref_x[i]) / ref_e_x[i])
        dis = work_width - (path_x[i] - ref_x[i]) / ref_e_x[i]
        if dis < 0:
            overlap.append(dis)
            skip.append(0)
        else:
            skip.append(dis)
            overlap.append(0)
    overlaps.append(sum(overlap) / len(overlap))
    skips.append(sum(skip) / len(skip))

    print(width)
    # calculating the actual offset direction of the waypoints
    angle_diff = []
    for i in range(Ng):
        normal_angle = math.atan2(ref_e_y[i], ref_e_x[i])
        offset_angle = math.atan2(path_y[i] - ref_y[i], path_x[i] - ref_x[i])
        angle_diff.append(offset_angle - normal_angle)
    offset_angle_diff.append(angle_diff)
    # 4.the second treatment to avoid self-intersection: remove the self-intersection sections
    # path_x_pro2 = path_x
    # path_y_pro2 = path_y
    path = LineString([(path_x[i], path_y[i]) for i in range(len(path_x))])
    if not path.is_simple:
        path_x_pro2, path_y_pro2 = rm_self_intersection(path_x, path_y)
    else:
        path_x_pro2 = path_x
        path_y_pro2 = path_y
    path_x_list.append(path_x_pro2)
    path_y_list.append(path_y_pro2)
    # 5.clipping the generated waylines according to the field boundary
    if id > 0:
        clipped_x, clipped_y = polygon_clip_line(path_x_pro2, path_y_pro2, inner_boundary_x, inner_boundary_y)
        print('clipped_x', clipped_x)
        if len(clipped_x) == 0:
            break
        else:
            if len(clipped_x) == 3:
                clipped_x.insert(1, (clipped_x[0] + clipped_x[1]) / 2)
                clipped_y.insert(1, (clipped_y[0] + clipped_y[1]) / 2)
                clipped_x.insert(-2, (clipped_x[-1] + clipped_x[-2]) / 2)
                clipped_y.insert(-2, (clipped_y[-1] + clipped_y[-2]) / 2)
            # 6.visualization
            if len(clipped_x) > 3:
                pic_x, pic_y, pic_u, pic_tck = generate_cubic_Bspline(clipped_x, clipped_y, pic_num)
                wayline_cur.append(calc_curvature(pic_u, pic_tck))
                plt.plot(pic_x, pic_y, 'g')
            else:
                plt.plot(clipped_x, clipped_y, 'g')


plt.axis('equal')
plt.show()

plt.figure()
plt.plot(init_u, [0] * 30)
plt.plot(init_u, offset_angle_diff[0], 'o')
# plt.axis('equal')
# plt.ylim([-0.01, 0.01])
plt.show()
