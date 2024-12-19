import sys
import joblib
import six
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, SVMSMOTE

from costcla.models import CostSensitiveRandomPatchesClassifier
import xgboost as xgb
from agga.tournament_performance import compute_pareto_front


sys.modules['sklearn.externals.joblib'] = joblib
sys.modules['sklearn.externals.six'] = six
sys.modules['sklearn.externals.six.moves'] = six.moves


class CappingModel:
    def __init__(self, scenario):
        self.ref_points = {}
        self.pareto_front = {}
        self.norm = True
        self.scenario = scenario

        model_params = {
                "rp": {
                "combination": "weighted_voting",
                "max_features": 0.7326478694061803,
                "max_samples": 0.1744647121719889,
                "n_estimators": 705,
                "pruned": False
                },

                "rf" : {
              'bootstrap': False,
              'max_depth': 3,
              'max_features': 0.13187903888451363,
              'min_samples_leaf': 17,
              'min_samples_split': 10,
              'n_estimators': 999,
                },

                "xgb":{
                "booster": "gbtree",
                "learning_rate": 0.002402132192436067,
                "n_estimators": 121,
                "reg_alpha": 8.798542765835673,
                "reg_lambda": 0.00149319499376181,
                "scale_pos_weight": 1.081926927217707,
                "colsample_bytree": 0.522423431423826,
                "gamma": 0.42064988876533926,
                "max_depth": 3,
                "min_child_weight": 3,
                "subsample": 0.5006162581582194
                }
            }


        if scenario.cap_model == "rf":
            self.model = RandomForestClassifier(**model_params[scenario.cap_model])
        elif scenario.cap_model == "rp":
            self.model = CostSensitiveRandomPatchesClassifier(**model_params[scenario.cap_model])
        elif scenario.cap_model == "xgb":
            self.model = xgb.XGBClassifier(**model_params[scenario.cap_model])

    def build_train_set(self, trajectories, cutoff):

        cutoff = [cutoff]
        # get the instances we have results for
        instances_training = []
        for conf, i_t in trajectories.items():
            if len(i_t) > len(instances_training):
                instances_training = list(i_t.keys())

        self.set_hv_ref_point(trajectories)
        conf_ids = list(trajectories.keys())

        # compute the pareto front for each instance
        for instance in instances_training:
            results_for_instance = {conf: trajectories[conf][instance] for conf in conf_ids if instance in trajectories[conf].keys()}
            conf_instance_results_full = np.array([point for conf_r in results_for_instance.values() for point in conf_r])
            self.pareto_front[instance] = compute_pareto_front(conf_instance_results_full, return_mask=False)

        features = []
        target = []
        cost_matrix = []
        # compute features, labels and costmatrix
        for co in cutoff:
            for conf, instances in trajectories.items():
                for instance, trajectory in instances.items():

                    # we only care about instances with results
                    if instance not in instances_training:
                        continue

                    # prep trajectories
                    if len(trajectory) != 0:
                        trajectory = self.drop_duplicates(trajectory)
                        trajectory = self.merge_points_same_timestep(trajectory)

                    trajectory_length_cut = [[p[0], p[1]] for p in trajectory if p[0] <= co]
                    features.append(self.compute_features(instance, trajectory_length_cut, co))
                    target.append(self.compute_label(trajectory, co))
                    cost, on_front = self.compute_cost_matrix(instance, co, trajectory)
                    cost_matrix.append(cost)

        return features, target, cost_matrix

    def compute_cost_matrix(self, instance, cutoff, trajectory):
        # for the cost we are only intrested in the solutions found after the potential cutoff
        trajectory_length_cut = [[p[0], p[1]] for p in trajectory if p[0] > cutoff]
        ind = IGD(np.array(self.pareto_front[instance]))
        max_point = np.array(self.ref_points[instance])
        max_distance = ind(np.array(max_point))
        on_font = False

        # compute distance
        if len(trajectory_length_cut) > 0 :
            distance = ind(np.array(trajectory_length_cut))
            distance_norm = (distance - 0) / (max_distance - 0)
            distance_norm = distance_norm

            if distance_norm == 0:
                on_font = True
        else:
            # If not solved take worst possible value times
            trajectory_length_cut = np.array(self.ref_points[instance])
            distance = ind(np.array(trajectory_length_cut))
            distance_norm = (distance - 0) / (max_distance - 0)
            distance_norm = distance_norm

        if self.norm:
            distance_norm = 1 - distance_norm
            return [distance_norm, distance_norm, 0 ,0], on_font
        else:
            return [distance, distance, 0, 0], on_font

    def fit(self, trajectories, cutoff):
        x, y, cost = self.build_train_set(trajectories, cutoff)
        x = np.array(x)
        y = np.array(y)
        cost = np.array(cost)

        # balance the data so we have the same amount of trajectories that do improve to the ones that do not
        x = np.c_[x, cost]

        smote = SVMSMOTE(k_neighbors=1, m_neighbors=1, out_step=0.74)
        x, y = smote.fit_resample(x, y)

        cost = x[:, -4:]
        x = x[:, :-4]

        if self.scenario.cap_model == "rp":
            self.model.fit(x, y,cost)
        else:
            self.model.fit(x, y)

    def predict_batch(self, trajectories, cutoff):
        # predict if we will be better after a time
        self.set_hv_ref_point(trajectories)

        x = []
        keys = []
        for conf, instances in trajectories.items():
            for instance, trajectory in instances.items():

                if len(trajectory) != 0:
                    trajectory = self.drop_duplicates(trajectory)
                    trajectory = self.merge_points_same_timestep(trajectory)

                x.append(self.compute_features(instance, trajectory, cutoff))
                keys.append([conf, instance])

        x = np.array(x)
        predictions = self.model.predict(x)

        keys_predictions = np.column_stack((keys, predictions))
        predictions_dic = {}
        for pred in keys_predictions:
            if pred[0] not in predictions_dic.keys():
                predictions_dic[pred[0]] = {}

            if pred[1] not in predictions_dic[pred[0]].keys():
                predictions_dic[pred[0]][pred[1]] = int(pred[2])
        return predictions_dic


    def compute_features(self,instance, trajectory, cutoff):
        features = []
        features = features + [self.feature_number_of_solutions(trajectory)]
        features = features + [self.feature_dif_start_end_quality(trajectory)]
        features = features + [self.feature_slope_mean(trajectory)]
        features = features + [self.feature_slope_first(trajectory)]
        features = features + [self.feature_slope_last(trajectory)]
        features = features + [self.feature_std_improvment(trajectory)]
        features = features + [self.feature_time_first_improvment(trajectory)]
        features = features + [self.feature_quality_first_improvment(trajectory)]
        features = features + [self.feature_time_last_improvment(trajectory)]
        features = features + [self.feature_quality_last_improvment(trajectory)]
        features = features + [self.feature_hv(trajectory, instance)]
        return features

    def compute_label(self, trajectory, cutoff):
        label = 0
        for p in trajectory:
            if p[0] > cutoff:
                label =  1
        return label

    def drop_duplicates(self, trajectory):
        # account for target algorithms printing the same quality multiple times at different timesteps
        # we only keep the first occurrences of the quality
        clean = []
        previous = trajectory[0][1] + 1
        for point in trajectory:
            if point[1] < previous:
                clean.append(point)
                previous = point[1]
        return clean

    def merge_points_same_timestep(self, trajectory):
        # account for target algorithms printing different quality at the same timestep.
        # keep the better quality
        clean = []
        previous = -1
        previous_p = trajectory[0]
        for point in trajectory:
            if point[0] > previous:
                clean.append(point)
                previous = point[0]
                previous_p = point
            elif point[0] == previous:
                clean.append(point)
                clean.remove(previous_p)
                previous = point[0]
                previous_p = point
        return clean

    def feature_number_of_solutions(self, trajectory):
        return len(trajectory)

    def feature_dif_start_end_quality(self, trajectory):
        if len(trajectory) > 1:
            return trajectory[0][1] - trajectory[-1][1]
        else:
            return 0

    def feature_slope_mean(self, trajectory):
        if len(trajectory) > 1:
            return (trajectory[-1][1] - trajectory[0][1]) / (trajectory[-1][0] - trajectory[0][0])
        else:
            return 0

    def feature_slope_first(self, trajectory):
        if len(trajectory) > 1:
            return (trajectory[1][1] - trajectory[0][1]) / (trajectory[1][0] - trajectory[0][0])
        else:
            return 0

    def feature_slope_last(self, trajectory):
        if len(trajectory) > 1:
            return (trajectory[-1][1] - trajectory[-2][1]) / (trajectory[-1][0] - trajectory[-2][0])
        else:
            return 0

    def feature_std_improvment(self, trajectory):
        if len(trajectory) > 1:
            return np.std(np.array([q[1] for q in trajectory]))
        else:
            return 0

    def feature_time_first_improvment(self, trajectory):
        if len(trajectory) > 0:
            return trajectory[0][0]
        else:
            return 0

    def feature_time_last_improvment(self, trajectory):
        if len(trajectory) > 0:
            return trajectory[-1][0]
        else:
            return 0

    def feature_quality_first_improvment(self, trajectory):
        if len(trajectory) > 0:
            return trajectory[0][1]
        else:
            return 0

    def feature_quality_last_improvment(self, trajectory):
        if len(trajectory) > 0:
            return trajectory[-1][1]
        else:
            return 0

    def set_hv_ref_point(self, trajectories):

        for conf, instances in trajectories.items():
            for instance, trajectory in instances.items():
                if instance not in self.ref_points.keys():
                    self.ref_points[instance] = [self.scenario.cutoff_time, -1]
                if len(trajectory) > 0:
                    if trajectory[0][1] > self.ref_points[instance][1]:
                        self.ref_points[instance] = [self.scenario.cutoff_time, trajectory[0][1]]


    def feature_hv(self, trajectory, instance):
        if len(trajectory) > 0:
            ref_point = self.ref_points[instance]
            ind_hv = HV(ref_point=[1, 1])
            full_pareto_front_norm = ((np.array(trajectory) - 0) / (np.array(ref_point)) - 0) * 0.9
            hv_full_front = ind_hv(full_pareto_front_norm)
            return hv_full_front
        else:
            return 0






