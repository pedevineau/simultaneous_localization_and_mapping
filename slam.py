import math

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl
import scipy.io

J = np.array([[0, -1],
              [1, 0]])


def is_meas_reliable(z, z_hat, S, thresh=20):
    return np.dot(np.dot(z - z_hat, np.linalg.inv(S)), (z - z_hat)) / 2. < thresh


def evaluate(traj_real, traj_est):
    leng = traj_est.shape[0]
    error = 0
    for k in range(len(traj_real)):
        error += np.linalg.norm(traj_real[0][:leng, 1:3] - traj_est[:, 3 * k:3 * k + 2])
    return error


def rot(theta):
    return np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)]
    ])


class Robot(object):
    # a robot is characterized by the list of its states [mu(t=0) mu(t=T), ... mu(t=nT)], and the list of sigma matrix
    def __init__(self, t0, x0=0, y0=0, th0=0):
        self.state = [x0, y0, th0]
        self.sig = np.zeros((3, 3))
        self.t = t0
        self.traj = []

    # m = list of landmarks' exact positions, z = list of current measurements
    def propagate_once(self, current_odom, dt, m, z, barcodes, without_observation):
        previous_state = self.state
        # TODO: we should check if current_odom[1] is in rad/s-1 or not
        th = previous_state[2]
        v = current_odom[0]
        w = current_odom[1]

        cos_th = np.cos(th)
        sin_th = np.sin(th)
        cos_fac = v * dt * cos_th
        sin_fac = v * dt * sin_th

        mu_bar_x = previous_state[0] + cos_fac
        mu_bar_y = previous_state[1] + sin_fac
        mu_bar_th = (th + w * dt)

        mu_bar = [mu_bar_x, mu_bar_y, mu_bar_th]

        # TODO: check G is the jacobian and not its transposed form
        G = np.array([[1, 0, -sin_fac], [0, 1, cos_fac], [0, 0, 1]])
        # TODO: check V is the jacobian and not its transposed form
        # V = dt*np.array([[cos_th, -0.5*sin_fac], [sin_th, 0.5*cos_fac], [0, 1]])
        V = np.array([
            [cos_th, -sin_th, 0],
            [sin_th, cos_th, 0],
            [0, 0, 1]
        ])

        # TODO: Q to be completed
        alphas = [0.2, 0.03, 0.1, 0.1]
        # Q = np.diag([alphas[0]*v**2 + alphas[1]*w**2, alphas[2]*v**2 + alphas[3]*w**2])
        # Q = np.diag([100,100])
        Q = np.diag([0.1 * abs(v), 0.01 * abs(v), 0.1 * abs(w)])

        sig = self.sig
        sig = np.dot(np.dot(G, sig), np.transpose(G)) + \
              np.dot(np.dot(V, Q), np.transpose(V))

        if not without_observation:
            H = []
            z_hat = []
            z_mes = []
            for i in range(len(z)):
                # if it is a landmark
                if z[i][1] in barcodes[5:]:
                    index = np.where(barcodes[5:] == z[i][1])
                    lmark_pos = m[index[0]][0]
                    dev_x = lmark_pos[0] - mu_bar_x
                    dev_y = lmark_pos[1] - mu_bar_y
                    q = dev_x ** 2 + dev_y ** 2
                    next_z = z[i][2:]
                    next_z_hat = np.array([math.sqrt(q), math.atan2(dev_y, dev_x) - mu_bar_th])
                    z_mes.append(next_z)
                    z_hat.append(next_z_hat)
                    H.append(np.array([[-dev_x / math.sqrt(q), -dev_y / math.sqrt(q), 0], [dev_y / q, -dev_x / q, -1]]))
                # else we are observing a robot
            if len(H) > 0:
                sigmas_vector = []
                for i in range(len(H)):
                    sigmas_vector.append(0.5 ** 2)
                    sigmas_vector.append((3 * math.pi / 180) ** 2)
                R = np.diag(sigmas_vector)
                Z = np.zeros((2 * len(H)))
                for k in range(len(H)):
                    Z[2 * k] = z_mes[k][0] - z_hat[k][0]
                    Z[2 * k + 1] = z_mes[k][1] - z_hat[k][1]

                # Final loop!
                # for i in range(len(H)):
                #         H_current = H[i]
                #         S = np.dot(np.dot(H_current, sig), np.transpose(H_current)) + R
                #         K = np.dot(np.dot(sig, np.transpose(H_current)), npl.inv(S))
                #         mu_bar = mu_bar + np.dot(K,(z_mes[i] - z_hat[i]))
                #         sig = np.dot(np.identity(3) - np.dot(K,H_current), sig)

                H_current = np.reshape(H, (2 * len(H), 3))
                # print H_current
                S = np.dot(np.dot(H_current, sig), np.transpose(H_current)) + R
                K = np.dot(np.dot(sig, np.transpose(H_current)), npl.inv(S))

                for k in range(len(Z) / 2):

                    if not is_meas_reliable(z_mes[k], z_hat[k], S[2 * k:2 * k + 2, 2 * k:2 * k + 2], 10):
                        K[:, 2 * k:2 * k + 2] = 0
                        H_current[2 * k:2 * k + 2, :] = 0
                        Z[2 * k:2 * k + 2] = 0

                mu_bar = mu_bar + np.dot(K, Z)
                sig = np.dot(np.identity(3) - np.dot(K, H_current), sig)

        self.state = mu_bar
        self.sig = sig
        self.t = self.t + dt

    def propagate(self, odom, dt, m, z, barcodes, ground, landmarks, without_observation):
        freq = 50
        cpt_z = 0
        if landmarks is not None:
            landx, landy = landmarks[:, 0], landmarks[:, 1]
        plot_traj = True
        for i in range(len(odom)):
            new_z = []
            if i % freq == 0:
                posx, posy = self.state[0], self.state[1]
                realx, realy = ground[i][1], ground[i][2]
            while cpt_z < len(z) and z[cpt_z][0] < self.t + dt:
                new_z.append(z)
                cpt_z += 1
            self.propagate_once(odom[i], dt, m, new_z, barcodes, without_observation)
            self.traj.append(self.state)

            if i % freq == 0 and i > 450:
                # print(i)
                if plot_traj:
                    plt.clf()
                    plt.scatter(posx, posy, c="b")
                    plt.scatter(realx, realy, c="r")
                    plt.scatter(landx, landy, c="g")
                    plt.show(block=False)
                    plt.pause(0.01)

    def plot_trajectory(self, odom, dt, m, z, ground, barcodes, landmarks, without_observation=False):
        self.propagate(odom, dt, m, z, barcodes, ground, landmarks, without_observation)
        self.traj = np.array(self.traj)
        X = self.traj[:, 0]
        Y = self.traj[:, 1]
        # Th = np.remainder(self.traj[:,2], 2*math.pi) -math.pi
        # ground_Th = np.remainder(ground[:,3], 2*math.pi) -math.pi
        # plt.plot(X)
        # plt.plot(Y)
        plt.plot(X, Y)
        # plt.plot(ground[:,0],Th)
        # plt.plot(ground[:,1])
        # plt.plot(ground[:,2])
        plt.plot(ground[:, 1], ground[:, 2])
        # plt.plot(odom[:,0])
        plt.show()


class Fleet(object):
    def __init__(self, t0, robots):
        n = len(robots)
        self.nrobots = n
        self.state = np.zeros(3 * n)
        self.sig = np.zeros((3 * n, 3 * n))
        for k in range(n):
            robot = robots[k]
            self.state[3 * k], self.state[3 * k + 1], self.state[3 * k + 2] = robot.state[0], robot.state[1], \
                                                                              robot.state[2]
            self.sig[3 * k:3 * k + 3, 3 * k:3 * k + 3] = robot.sig
        self.t = t0
        self.traj = []

    def propagate_once(self, current_odom, dt, m, z, barcodes, ground=None):
        previous_state = self.state
        G = np.zeros((3 * self.nrobots, 3 * self.nrobots))
        V = np.zeros((3 * self.nrobots, 3 * self.nrobots))
        Q = np.zeros((3 * self.nrobots, 3 * self.nrobots))
        mu_bar = []
        # propagate
        for k in range(self.nrobots):
            th = previous_state[3 * k + 2]
            v = current_odom[2 * k]
            w = current_odom[2 * k + 1]
            cos_th = np.cos(th + w * dt / 2)
            sin_th = np.sin(th + w * dt / 2)
            cos_fac = v * dt * cos_th
            sin_fac = v * dt * sin_th
            mu_bar.append(previous_state[3 * k] + cos_fac)
            mu_bar.append(previous_state[3 * k + 1] + sin_fac)
            mu_bar.append(th + w * dt)
            G[3 * k:3 * k + 3, 3 * k:3 * k + 3] = np.array([[1, 0, -sin_fac], [0, 1, cos_fac], [0, 0, 1]])

            V[3 * k:3 * k + 3, 3 * k:3 * k + 3] = np.array([[cos_th, -sin_th, 0], [sin_th, cos_th, 0], [0, 0, 1]])
            alphas = [2, 3, 10, 10]
            # Q[2*k:2*k+2, 2*k:2*k+2] = np.diag([alphas[0]*v**2 + alphas[1]*w**2, alphas[2]*v**2 + alphas[3]*w**2])
            Q[3 * k:3 * k + 3, 3 * k:3 * k + 3] = np.diag([0.1 * abs(v), 0.01 * abs(v), 0.1 * abs(w)])

            # Q[2*k:2*k+2, 2*k:2*k+2] = np.diag([0.1, 0.1])

        nb_obs = 0
        for k in range(len(z)):
            nb_obs += len(z[k])

        z_mes = []
        z_hat = []
        # H = np.zeros((2*nb_obs, 3*self.nrobots))
        H = []

        count_identified_obs = 0

        for k in range(self.nrobots):
            break
            obs = z[k]  # may be empty
            for i in range(len(obs)):
                # if it is a landmark
                if obs[i][1] in barcodes[5:]:
                    index = np.where(barcodes[5:] == obs[i][1])
                    lmark_pos = m[index[0]][0]
                    dev_x = lmark_pos[0] - mu_bar[3 * k]
                    dev_y = lmark_pos[1] - mu_bar[3 * k + 1]
                    q = dev_x ** 2 + dev_y ** 2
                    z_mes.append(obs[i][2:])
                    z_hat.append(np.array([math.sqrt(q), math.atan2(dev_y, dev_x) - mu_bar[3 * k + 2]]))

                    Hi = np.zeros((2, 3 * self.nrobots))
                    Hi[:, 3 * k:3 * k + 3] = [[-dev_x / math.sqrt(q), -dev_y / math.sqrt(q), 0],
                                              [dev_y / q, -dev_x / q, -1]]
                    count_identified_obs += 1
                # else we are observing a robot
                elif obs[i][1] in barcodes[:5]:
                    # assuming the index of Robot1 is 0,...
                    observed_robot = np.where(barcodes[:5] == obs[i][1])[0][0]
                    dev_x = mu_bar[3 * observed_robot] - mu_bar[3 * k]
                    dev_y = mu_bar[3 * observed_robot + 1] - mu_bar[3 * k + 1]

                    q = dev_x ** 2 + dev_y ** 2
                    z_mes.append(obs[i][2:])
                    z_hat.append(np.array([math.sqrt(q), math.atan2(dev_y, dev_x) - mu_bar[3 * k + 2]]))

                    Hi = np.zeros((2, 3 * self.nrobots))
                    Hi[:, 3 * k:3 * k + 3] = [[-dev_x / math.sqrt(q), -dev_y / math.sqrt(q), 0],
                                              [dev_y / q, -dev_x / q, -1]]
                    Hi[:, 3 * observed_robot:3 * observed_robot + 3] = \
                        np.array([[dev_x / math.sqrt(q), dev_y / math.sqrt(q), 0], [-dev_y / q, dev_x / q, 0]])

                    count_identified_obs += 1

                else:
                    break
                try:
                    H = np.concatenate((H, Hi), axis=0)
                except:
                    H = Hi

        H = np.array(H)
        nb_obs = count_identified_obs
        sig = self.sig
        sig = np.dot(np.dot(G, sig), np.transpose(G)) + \
              np.dot(np.dot(V, Q), np.transpose(V))

        if nb_obs > 0:
            sigmas_vector = []
            for i in range(nb_obs):
                sigmas_vector.append(0.5 ** 2)
                sigmas_vector.append((3 * math.pi / 180) ** 2)
            R = np.diag(sigmas_vector)
            Z = np.zeros((2 * nb_obs))
            for k in range(nb_obs):
                Z[2 * k] = z_mes[k][0] - z_hat[k][0]
                Z[2 * k + 1] = z_mes[k][1] - z_hat[k][1]

            # Final loop!
            # for i in range(len(H)):
            #         H_current = H[i]
            #         S = np.dot(np.dot(H_current, sig), np.transpose(H_current)) + R
            #         K = np.dot(np.dot(sig, np.transpose(H_current)), npl.inv(S))
            #         mu_bar = mu_bar + np.dot(K,(z_mes[i] - z_hat[i]))
            #         sig = np.dot(np.identity(3) - np.dot(K,H_current), sig)

            # print H_current
            S = np.dot(np.dot(H, sig), np.transpose(H)) + R
            K = np.dot(np.dot(sig, np.transpose(H)), npl.inv(S))
            mu_bar = mu_bar + np.dot(K, Z)
            sig = np.dot(np.identity(3 * self.nrobots) - np.dot(K, H), sig)

        self.state = mu_bar
        self.sig = sig
        self.t = self.t + dt

    def propagate(self, odom, dt, m, z, barcodes, ground=None):
        cpt_z = np.zeros(self.nrobots, dtype=int)
        for i in range(len(odom)):
            new_z = []
            for k in range(self.nrobots):
                new_obs = []
                while cpt_z[k] < len(z[k]) and z[k][cpt_z[k]][0] < self.t + dt:
                    new_obs.append(z[k][cpt_z[k]])
                    cpt_z[k] += 1
                new_z.append(new_obs)
            self.propagate_once(odom[i], dt, m, new_z, barcodes, ground)
            self.traj.append(self.state)
        self.traj = np.array(self.traj)

    def plot_trajectory(self, robot_number, odom, dt, m, z, ground, barcodes):
        self.propagate(odom, dt, m, z, barcodes, ground)
        X = self.traj[:, 3 * robot_nb]
        Y = self.traj[:, 3 * robot_nb + 1]
        # Th = np.remainder(self.traj[:,2], 2*math.pi) -math.pi
        # ground_Th = np.remainder(ground[:,3], 2*math.pi) -math.pi
        plt.plot(X)
        plt.plot(Y)
        # plt.plot(X,Y)
        # plt.plot(ground[:,0],Th)
        plt.plot(ground[robot_nb][:, 1])
        plt.plot(ground[robot_nb][:, 2])
        # plt.plot(ground[robot_nb][:,1], ground[robot_nb][:,2])
        plt.show()


class Slam(object):
    def __init__(self, t0, fleet, nb_landmarks, barcodes):
        # self.state = np.concatenate((fleet.state, np.zeros(nb_landmarks*2)))
        self.t = t0
        self.nrobots = fleet.nrobots
        self.nlandmarks = nb_landmarks
        N = 3 * self.nrobots + 2 * self.nlandmarks
        self.N = N
        self.state = np.zeros(self.N)
        self.state[:3 * self.nrobots] = np.copy(fleet.state)
        self.sig = np.zeros((N, N))
        portfolio = dict()
        for k in range(5, len(barcodes)):
            portfolio[str(barcodes[k][1])] = 3 * self.nrobots + 2 * (k - 5)
        self.portfolio = portfolio
        self.sig[:3 * self.nrobots, :3 * self.nrobots] = fleet.sig
        self.traj = []

    def propagate_once(self, current_odom, dt, z, barcodes, ground=None, std_r=0.5, std_phi_deg=3, std_v=0.1,
                       std_w=0.1):
        previous_state = np.copy(self.state)
        G = np.zeros((self.N, self.N))
        V = np.zeros((self.N, 3 * self.nrobots))
        Q = np.zeros((3 * self.nrobots, 3 * self.nrobots))
        mu_bar = []
        # propagate
        for k in range(self.nrobots):
            th = previous_state[3 * k + 2]
            v = current_odom[2 * k]
            w = current_odom[2 * k + 1]
            cos_th = np.cos(th + w * dt / 2)
            sin_th = np.sin(th + w * dt / 2)
            cos_fac = v * dt * cos_th
            sin_fac = v * dt * sin_th
            mu_bar.append(previous_state[3 * k] + cos_fac)
            mu_bar.append(previous_state[3 * k + 1] + sin_fac)
            mu_bar.append(th + w * dt)
            G[3 * k:3 * k + 2, 3 * k + 2] = J.dot(rot(th)).dot(np.array([v, 0]))
            V[3 * k:3 * k + 3, 3 * k:3 * k + 3] = np.array([[cos_th, -sin_th, 0], [sin_th, cos_th, 0], [0, 0, 1]])
            Q[3 * k:3 * k + 3, 3 * k:3 * k + 3] = np.diag([std_v * abs(v), std_v * 0.05 * abs(v), std_w * abs(w)])

        for k in range(self.nlandmarks):
            mu_bar.append(self.state[3 * self.nrobots + 2 * k])
            mu_bar.append(self.state[3 * self.nrobots + 2 * k + 1])

        sig = self.sig
        Psi = np.eye(self.N) + G * dt + 1 / 2 * G.dot(G) * dt ** 2 + 1 / 6 * G.dot(G).dot(G) * dt ** 3
        Q_dt = Psi.dot(V).dot(Q).dot(V.T).dot(Psi.T) * dt
        self.sig = Psi.dot(self.sig).dot(Psi.T) + Q_dt
        self.state = np.array(mu_bar)
        ##################

        # update
        nb_obs = 0
        for k in range(len(z)):
            nb_obs += len(z[k])

        z_mes = []
        z_hat = []
        H = []

        count_identified_obs = 0

        for k in range(self.nrobots):
            obs = z[k]  # may be empty
            len_obs = len(obs) - 1
            for i in range(len_obs, -1, -1):
                # if it is a landmark
                if obs[i][1] in barcodes[5:, 1] and True:
                    coordinate_ldmark = self.portfolio[str(int(obs[i][1]))]
                    lmark_known_x = self.state[coordinate_ldmark]
                    lmark_known_y = self.state[coordinate_ldmark + 1]
                    r = obs[i][2]
                    phi = obs[i][3]
                    angle = self.state[3 * k + 2] + phi
                    lmark_x = self.state[3 * k] + r * math.cos(angle)
                    lmark_y = self.state[3 * k + 1] + r * math.sin(angle)
                    # initialize
                    if lmark_known_x == 0 and lmark_known_y == 0:
                        self.state[coordinate_ldmark] = lmark_x
                        self.state[coordinate_ldmark + 1] = lmark_y
                        mini_R = np.diag([0.5 ** 2, (3 * 3.14 / 180) ** 2])
                        Rot = rot(self.state[3 * k + 2])
                        pos_robot2landmark = r * np.array([math.cos(phi), math.sin(phi)])
                        H_grad = np.array([1 / np.linalg.norm(pos_robot2landmark) * pos_robot2landmark,
                                           1 / (np.linalg.norm(pos_robot2landmark) ** 2) * pos_robot2landmark.dot(J.T)])
                        I_2 = np.eye(2)
                        mat_aux = J.dot(np.array([[lmark_x - self.state[3 * k]], [lmark_y - self.state[3 * k + 1]]]))
                        H_robot = -H_grad.dot(Rot.T).dot(np.concatenate((I_2, mat_aux), axis=1))
                        H_landmark = H_grad.dot(Rot.T)
                        H_il = np.linalg.inv(H_landmark)
                        P_robot = self.sig[3 * k:3 * k + 3, 3 * k:3 * k + 3]

                        # sigma (L, L)
                        self.sig[coordinate_ldmark:coordinate_ldmark + 2, coordinate_ldmark:coordinate_ldmark + 2] = \
                            H_il.dot(H_robot).dot(P_robot).dot(H_robot.T).dot(H_il.T) + \
                            H_il.dot(mini_R).dot(H_il.T)

                        # sigma (R, L)
                        self.sig[coordinate_ldmark:coordinate_ldmark + 2, 3 * k:3 * k + 3] = \
                            -H_il.dot(H_robot).dot(P_robot)

                        self.sig[3 * k:3 * k + 3, coordinate_ldmark:coordinate_ldmark + 2] = \
                            self.sig[coordinate_ldmark:coordinate_ldmark + 2, 3 * k:3 * k + 3].T
                        # print('Landmark init', obs[i][1], self.state[coordinate_ldmark])
                        del obs[i]

        for k in range(self.nrobots):
            obs = z[k]  # may be empty
            for i in range(len(obs)):
                if obs[i][1] in barcodes[5:, 1] and True:
                    coordinate_ldmark = self.portfolio[str(int(obs[i][1]))]
                    dev_x = self.state[coordinate_ldmark] - self.state[3 * k]
                    dev_y = self.state[coordinate_ldmark + 1] - self.state[3 * k + 1]
                    q = dev_x ** 2 + dev_y ** 2
                    z_mes.append(obs[i][2:])
                    x2x_measured = rot(self.state[3 * k + 2]).T.dot(np.array([dev_x, dev_y]))
                    z_hat.append(np.array([math.sqrt(q), math.atan2(x2x_measured[1], x2x_measured[0])]))
                    DH = np.array([
                        (1 / np.linalg.norm(x2x_measured) * x2x_measured),
                        (1 / np.linalg.norm(x2x_measured) ** 2) * x2x_measured.dot(J.T)
                    ])
                    matr = np.zeros((2, 3))
                    matr[0, 0] = 1
                    matr[1, 1] = 1
                    matr[:, 2] = J.dot(np.array([dev_x, dev_y]))

                    Hi = np.zeros((2, self.N))
                    Hi[:, 3 * k:3 * k + 3] = -DH.dot(rot(self.state[3 * k + 2]).T).dot(matr)

                    Hi[:, coordinate_ldmark: coordinate_ldmark + 2] = DH.dot(rot(self.state[3 * k + 2]).T)

                    count_identified_obs += 1
                # else we are observing a robot
                elif obs[i][1] in barcodes[:5] and False:
                    # assuming the index of Robot1 is 0,...
                    observed_robot = np.where(barcodes[:5] == obs[i][1])[0][0]
                    dev_x = self.state[3 * observed_robot] - self.state[3 * k]
                    dev_y = self.state[3 * observed_robot + 1] - self.state[3 * k + 1]

                    q = dev_x ** 2 + dev_y ** 2
                    z_mes.append(obs[i][2:])
                    z_hat.append(np.array([math.sqrt(q), math.atan2(dev_y, dev_x) - self.state[3 * k + 2]]))

                    x2x_measured = rot(self.state[3 * k + 2]).T.dot(np.array([dev_x, dev_y]))
                    DH = np.array([
                        (1 / np.linalg.norm(x2x_measured) * x2x_measured),
                        (1 / np.linalg.norm(x2x_measured) ** 2) * x2x_measured.dot(J.T)
                    ])
                    matr = np.zeros((2, 3))
                    matr[0, 0] = 1
                    matr[1, 1] = 1
                    matr[:, 2] = J.dot(np.array([dev_x, dev_y]))

                    Hi = np.zeros((2, self.N))
                    Hi[:, 3 * k:3 * k + 3] = -DH.dot(rot(self.state[3 * k + 2]).T).dot(matr)

                    Hi[:, 3 * observed_robot:3 * observed_robot + 2] = DH.dot(rot(self.state[3 * k + 2]))

                    count_identified_obs += 1

                else:
                    break
                if not len(H) == 0:
                    H = np.concatenate((H, Hi), axis=0)
                else:
                    H = Hi

        H = np.array(H)
        nb_obs = count_identified_obs

        if nb_obs > 0:
            sigmas_vector = []
            for i in range(nb_obs):
                sigmas_vector.append(std_r ** 2)
                sigmas_vector.append((math.pi * std_phi_deg / 180) ** 2)
            R = np.diag(sigmas_vector)
            Z = np.zeros((2 * nb_obs))
            for k in range(nb_obs):
                Z[2 * k] = z_mes[k][0] - z_hat[k][0]
                Z[2 * k + 1] = z_mes[k][1] - z_hat[k][1]

            # print H_current
            sig = self.sig
            S = np.dot(np.dot(H, sig), np.transpose(H)) + R
            K = np.dot(np.dot(sig, np.transpose(H)), npl.inv(S))

            for k in range(H.shape[0] / 2):
                if not is_meas_reliable(z_mes[k], z_hat[k], S[2 * k:2 * k + 2, 2 * k:2 * k + 2]):
                    K[:, 2 * k:2 * k + 2] = 0
                    H[2 * k:2 * k + 2, :] = 0
                    Z[2 * k:2 * k + 2] = 0

            self.state = self.state + np.dot(K, Z)
            sig = np.dot(np.identity(self.N) - np.dot(K, H), sig)
        # print(np.linalg.norm(sig))
        self.sig = sig
        self.t = self.t + dt

    def propagate(self, odom, dt, z, barcodes, ground=None, landmarks=None,
                  std_r=0.5, std_phi_deg=3, std_v=0.1, std_w=0.1, return_evaluate=True, plot_traj=False):
        freq = 50
        if landmarks is not None:
            landx, landy = landmarks[:, 0], landmarks[:, 1]
        est_landx, est_landy = np.zeros(15), np.zeros(15)
        cpt_z = np.zeros(self.nrobots, dtype=int)
        posx, posy = np.zeros(5), np.zeros(5)
        realx, realy = np.zeros(5), np.zeros(5)
        for i in range(len(odom)):
            new_z = []
            for k in range(self.nrobots):
                if i % freq == 0:
                    posx[k], posy[k] = self.state[3 * k], self.state[3 * k + 1]
                    realx[k], realy[k] = ground[k][i][1], ground[k][i][2]
                new_obs = []
                while cpt_z[k] < len(z[k]) and z[k][cpt_z[k]][0] < self.t + dt:
                    new_obs.append(z[k][cpt_z[k]])
                    cpt_z[k] += 1
                new_z.append(new_obs)
            self.propagate_once(odom[i], dt, new_z, barcodes, ground, std_r, std_phi_deg, std_v, std_w)
            self.traj.append(self.state)
            if i % freq == 0 and i > 450:
                for j in range(15):
                    est_landx[j], est_landy[j] = self.state[3 * self.nrobots + 2 * j], self.state[
                        3 * self.nrobots + 2 * j + 1]
                # print(i)
                if plot_traj:
                    plt.clf()
                    plt.scatter(posx, posy, c="b")
                    plt.scatter(realx, realy, c="r")
                    plt.scatter(est_landx, est_landy, c="y")
                    if landmarks is not None:
                        plt.scatter(landx, landy, c="g")
                    plt.show(block=False)
                    plt.pause(0.01)
                # print "time, evaluation", i/50, evaluate(ground, np.array(self.traj))
            if return_evaluate and i == 200 * 50:
                return evaluate(ground, np.array(self.traj))
        self.traj = np.array(self.traj)

    def plot_trajectory(self, robot_nb, odom, dt, z, ground, barcodes, landmarks=None,
                        std_r=0.5, std_phi_deg=3, std_v=0.1, std_w=0.1):
        self.propagate(odom, dt, z, barcodes, ground, landmarks, std_r, std_phi_deg, std_v, std_w,
                       return_evaluate=False, plot_traj=True)

        X = self.traj[:, 3 * robot_nb]
        Y = self.traj[:, 3 * robot_nb + 1]
        # Th = np.remainder(self.traj[:,2], 2*math.pi) -math.pi
        # ground_Th = np.remainder(ground[:,3], 2*math.pi) -math.pi
        # plt.plot(X)
        # plt.plot(Y)
        plt.plot(X, Y)
        # plt.plot(ground[:,0],Th)
        # plt.plot(ground[:,1])
        # plt.plot(ground[:,2])
        plt.plot(ground[robot_nb][:, 1], ground[robot_nb][:, 2])
        plt.show()


def prepare_data():
    return


if __name__ == '__main__':

    series_nb = 8
    path = 'data/MRCLAM' + str(series_nb) + '.mat'
    data = scipy.io.loadmat(path)
    barcodes = data['Barcodes']
    landmarks = data['Landmark_Groundtruth'][:, 1:3]

    odom = []
    meas = []
    ground = []
    list_robots = []
    nb_rob = 5

    for robot_nb in range(nb_rob):
        odom_str = 'Robot' + str(robot_nb + 1) + '_Odometry'
        meas_str = 'Robot' + str(robot_nb + 1) + '_Measurement'
        ground_str = 'Robot' + str(robot_nb + 1) + '_Groundtruth'
        odom.append(data[odom_str])
        meas.append(data[meas_str])
        ground.append(data[ground_str])
        list_robots.append(
            Robot(odom[robot_nb][0][0], ground[robot_nb][0][1], ground[robot_nb][0][2], ground[robot_nb][0][3]))
    fleet = Fleet(odom[0][0][0], list_robots)

    odom_final = []
    for t in range(len(odom[0])):
        genuine_od = []
        for rob in range(nb_rob):
            genuine_od.append(odom[rob][t][1])
            genuine_od.append(odom[rob][t][2])
        odom_final.append(np.array(genuine_od))
    gcm = Slam(odom[0][0][0], fleet, 15, barcodes)

    # ex demo: std_phi_deg=0.5 or 5
    gcm.propagate(odom_final, 0.02, meas, barcodes, ground, landmarks, return_evaluate=False, plot_traj=True,
                  std_phi_deg=0.5)

    raise (Exception)

    display_robot = 0

    minim_error = 10000000
    nb_iter = 0

    for std_r in (0.5 * np.linspace(0.4, 2, num=5, endpoint=False)):
        for std_phi_deg in (5 * np.linspace(0.4, 2, num=5, endpoint=False)):
            for std_v in (0.1 * np.linspace(0.4, 2, num=5, endpoint=False)):
                for std_w in (0.1 * np.linspace(0.4, 2, num=5, endpoint=False)):
                    nb_iter += 1
                    gcm = GrandCorpsMalade(odom[robot_nb][0][0], fleet, 15, barcodes)
                    err = gcm.propagate(odom_final, 0.02, meas, barcodes, ground, landmarks,
                                        std_r, std_phi_deg, std_v, std_w)
                    if err < minim_error:
                        minim_error = err
                        std_phi_deg_best = std_phi_deg
                        std_r_best = std_r
                        std_v_best = std_v
                        std_w_best = std_w
                    print "iter:", nb_iter, " , error:", err, " , min error:", minim_error

    print "min error", minim_error, " , std r:", std_r_best, " , std phi:", std_phi_deg_best, \
        " , std v:", std_v_best, " , std w:", std_w_best

    # gcm.plot_trajectory(display_robot, odom_final, 0.02, meas, ground, barcodes, landmarks,
    #                     std_r=0.5, std_phi_deg=5, std_v=0.1, std_w=0.1)
