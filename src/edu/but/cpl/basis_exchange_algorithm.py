from enum import Enum
import numpy as np # type: ignore
from edu.but.cpl.consts import INFINITY, ZERO, EQUALS_EPSILON, EQUAL_ZERO


class ProcedureState(Enum):
    NORMAL_STATE = 1
    BEGIN_DEGENERATION_STATE = 2
    DEGENERATION_STATE = 3
    END_DEGENERATION_STATE = 4



class BasisExchangeAlgorithm:

    def __init__(
        self,
        feature_vectors: np.ndarray,
        sample_weight: np.ndarray,
        feature_weight: np.ndarray,
        p_lambda: float,
        allow_feature_vectors_exit: bool = True,
        allow_feature_vectors_enter: bool = True,
        deep_copy: bool = False
    ):
        self.__feature_vectors_fullsize = feature_vectors.copy() if deep_copy else feature_vectors
        self.__sample_weight = sample_weight.copy() if deep_copy else sample_weight
        self.__feature_weight_fullsize = feature_weight.copy() if deep_copy else feature_weight
        self.__p_lambda = p_lambda
        self.__allow_feature_vectors_exit = allow_feature_vectors_exit
        self.__allow_feature_vectors_enter = allow_feature_vectors_enter

    
    @property
    def hyperplane(self) -> dict:
        """
        Return information about the current separating hyperplane.

        Returns
        -------
        hyperplane : dict
            Dictionary with keys 'features', 'coefs', 'theta'.
        """
        real_features = np.nonzero(~self.__uv_in_base)[0]
        theta_index = None
        if (self.__theta_index is not None) and (self.__theta_index in self.__features):
            theta_index = np.where(self.__features == self.__theta_index)[0][0]
            real_features = real_features[real_features != theta_index]
        return {
            "features": self.__features[real_features],
            "coefs": self.__vertex[real_features],
            "theta": self.__vertex[theta_index] if (theta_index is not None) and (self.__uv_in_base[theta_index] == False) else 0.
        }
    

    @property
    def criterion_function_value(self) -> float:
        """
        Return the value of the criterion function for the current vertex.

        Returns
        -------
        criterion_function_value : float
            The value of the criterion function.
        """
        return sum(self.__func_crit(self.__get_current_vertex()))


    def reset_procedure(self, features: np.ndarray = None, theta_index: int = None):
        """
        Prepare procedure for the optimisation process in a given feature subspace.

        Parameters
        ----------
        features : np.ndarray, default None
            Indexes of features to be used in the optimisation procedure. If None use all features.
        theta_index : int, default None
            Value from features of theta feature. If None not use theta. 
        """
        self.__features = features or np.arange(self.__feature_vectors_fullsize.shape[1])
        self.__theta_index = theta_index
        self.__feature_vectors = self.__feature_vectors_fullsize[:,self.__features]
        self.__feature_weight = self.__feature_weight_fullsize[self.__features]

        self.__num, self.__dim = self.__feature_vectors.shape

        self.__B_type = np.full((self.__dim), False)
        self.__B_index = np.arange(self.__dim)
        self.__B1 = np.eye(self.__dim)
        self.__fv_in_base = np.full((self.__num), False)
        self.__uv_in_base = np.full((self.__dim), True)

        self.__vertex = np.zeros(self.__dim)
        self.__cv = np.zeros(self.__dim)
        self.__fv_on_positive_site = np.full((self.__num), True)
        self.__ev = np.full((self.__dim), 1.)
        self.__products_fv_vertex = np.zeros(self.__num)
        self.__products_fv_B1l = np.zeros(self.__num)

        self.__procedure_state = ProcedureState.NORMAL_STATE
        self.__seq = None
        self.__epsilon_neighbourhood = None
        self.__l = None
        self.__lv = None
        self.__hold_direction = None
        self.__kv = None
        self.__s = None


    def add_feature(self, feature_index: int):
        """
        Extension of the current feature space with a feature with index feature_index.

        Parameters
        ----------
        feature_index : int
            The index of the feature by which the current feature space is to be extended.
        """
        if feature_index in self.__features:
            raise ValueError(f"The feature with index {feature_index} is already in the current feature space.")
        self.__features = np.append(self.__features, feature_index)
        self.__feature_weight = np.append(self.__feature_weight, self.__feature_weight_fullsize[feature_index])


    def optimize(self, verbose: int = 0):
        """
        Determine the optimal separating hyperplane (self.__vertex) for the current data and specified conditions.

        Parameters
        ----------
        verbose : int, default 0
            Refers to process logging, 0 means no logging, 1 means show informations during processing.
        """
        # correction_vector
        self.__cv = (self.__feature_vectors * np.array([self.__sample_weight]).T).sum(axis=0)
        self.__cv -= self.__p_lambda * self.__ev * self.__feature_weight

        n_step = 0
        while True:
            n_step += 1
            self.__print_log(verbose, n_step)
            if not self.__find_exit_edge():
                break
            if not self.__find_new_base_vector():
                break
            if self.__procedure_state == ProcedureState.NORMAL_STATE:
                if self.__check_degeneration():
                    print("DEGENERACJA")
            self.__update_to_new_vertex()

        if self.__procedure_state == ProcedureState.DEGENERATION_STATE:
            self.__optimal_vertex()
            self.__procedure_state = ProcedureState.NORMAL_STATE        
        self.__print_log(verbose, n_step)


    def __find_exit_edge(self) -> bool:
        """
        Determine the vector to exit the base.
        After completing the procedure in self.__l, the index of the row in the base containing the vector leaving
        the database is stored.

        Returns
        -------
        state : bool
            True - The vector for leaving the base has been determined.
            False - There is no candidate for leaving the base.
        """
        cv_projections = self.__cv @ self.__B1
        # cv_projections = self.__cv[~self.__uv_in_base] @ self.__B1[~self.__uv_in_base]
        # B_uv_row_ids = np.nonzero(~self.__B_type)   # ids of rows in base with unit vectors
        # B_uv_ids = self.__B_index[~self.__B_type]   # ids of unit vectors in base
        # cv_projections[B_uv_row_ids] += self.__cv[B_uv_ids] * self.__ev[B_uv_ids]

        if self.__allow_feature_vectors_exit:
            fv_selection = self.__B_type & (cv_projections < 0)
            cv_projections[fv_selection] *= -1 
            cv_projections[fv_selection] -= self.__sample_weight[self.__B_index[fv_selection]]
        else:
            fv_selection = self.__B_type
            cv_projections[fv_selection] = -INFINITY

        ev_selection = (~self.__B_type) & (cv_projections < 0)
        cv_projections[ev_selection] *= -1
        cv_projections[ev_selection] -= 2 * self.__p_lambda * self.__feature_weight[self.__B_index[ev_selection]]

        # def v_info(v):
        #     if v is None:
        #         return "None"
        #     if v[0]:
        #         return f"{v[1]}d"
        #     else:
        #         return f"{v[1]}{('-','+')[int(self.__ev[v[1]]==1)]}e"
        # B_desc = [v_info((t,i)) for t,i in zip(self.__B_type, self.__B_index)]
        # print("\n--> projections:", sorted(list(zip(B_desc, cv_projections, self.__cv @ self.__B1)), key=lambda x:x[1], reverse=True))

        if len(cv_projections) == 0:
            return False
        l = np.argmax(cv_projections)
        if cv_projections[l] <= ZERO:
            return False
        self.__l = l
        return True
    

    def __find_new_base_vector(self) -> bool:
        """
        Move along the exit edge to the new vector that is to enter the base.
        After completing the procedure in self.__kv, the data of the vector entering the base is stored.

        Returns
        -------
        state : bool
            True - There has been a shift to a new vector, which will enter the database.
            False - No exit edge crossings were found.
        """
        self.__init_movement_by_exit_edge()
        if self.__s < ZERO/1e3:
            return False
        if not self.__find_sequence_on_exit_edge():
            if self.__procedure_state != ProcedureState.DEGENERATION_STATE:
                return False
            return self.__end_degeneration()
        else:
            self.__go_to_new_vertex()
            gradient_zeroed = (self.__s <= 0)
            if not gradient_zeroed and self.__procedure_state == ProcedureState.DEGENERATION_STATE:
                return self.__end_degeneration()
            return True
        

    def __init_movement_by_exit_edge(self) -> None:
        """
        Initiate the process of moving along the exit edge.
        In self.__s, set the initial gradient value at the exit edge.
        """
        self.__lv = (self.__B_type[self.__l], self.__B_index[self.__l])
        cv_projection_l = self.__cv @ self.__B1[:,self.__l]
        self.__hold_direction = (cv_projection_l >= 0)
        if self.__hold_direction:
            self.__s = cv_projection_l
        else:
            self.__s = -cv_projection_l
            if self.__lv[0]:
                self.__s -= self.__sample_weight[self.__lv[1]]
                self.__modify_cv_minus(self.__lv[1])
            else:
                self.__s -= 2 * self.__p_lambda * self.__feature_weight[self.__lv[1]]
                self.__modify_cv_unit(self.__lv[1])


    def __find_sequence_on_exit_edge(self) -> bool:
        """
        Determine feature vectors and unit vectors on exit edge.
        In self.__seq, the vectors crossing of the exit edge are stored.

        Returns
        -------
        state : bool
            True - At least one crossings of the exit edge has been found.
            False - No exit edge crossings were found.
        """
        self.__seq = []
        if self.__procedure_state == ProcedureState.NORMAL_STATE:
            if self.__allow_feature_vectors_enter:
                self.__seq += self.__feature_vectors_on_exit_edge()
            self.__seq += self.__unit_vectors_on_exit_edge()
        elif self.__procedure_state == ProcedureState.DEGENERATION_STATE:
            if self.__allow_feature_vectors_enter:
                idxs_tbc = [v[1] for v in self.__epsilon_neighbourhood if v[0]]
                self.__seq += self.__feature_vectors_on_exit_edge(idxs_tbc, True)
            idxs_tbc = [v[1] for v in self.__epsilon_neighbourhood if not v[0]]
            self.__seq += self.__unit_vectors_on_exit_edge(idxs_tbc, True)
        elif self.__procedure_state == ProcedureState.END_DEGENERATION_STATE:
            if self.__allow_feature_vectors_enter:
                idxs_tbc = set(range(self.__num)) - set([v[1] for v in self.__epsilon_neighbourhood if v[0]])
                self.__seq += self.__feature_vectors_on_exit_edge(idxs_tbc)
            idxs_tbc = set(range(self.__dim)) - set([v[1] for v in self.__epsilon_neighbourhood if not v[0]])
            self.__seq += self.__unit_vectors_on_exit_edge(idxs_tbc)

        if self.__lv[0]:
            self.__products_fv_B1l[self.__lv[1]] = self.__feature_vectors[self.__lv[1]] @ self.__B1[:,self.__l]
            if not self.__hold_direction:
                self.__products_fv_B1l[self.__lv[1]] *= -1
            # if not self.allow_feature_vectors_enter:
            #     self.fvs_mask[self.lv[1]] = False

        return len(self.__seq) > 0


    def __feature_vectors_on_exit_edge(self, idxs_tbc=None, spread_edges=False):
        """
        Parameters
        ----------
        idxs_tbc : Iterable[int], default None
            The list of indices of feature vectors to be considered as candidates to appear on the output edge
        spread_edges : bool, default False
            Whether to spread the edges crossing the exit edge, needed for degeneration

        Returns
        -------
        fv_seq : list[tuple[bool,int,float]]
            Information about features vectors found on exit edge.
        """
        if idxs_tbc is None:
            idxs_tbc = np.nonzero(~self.__fv_in_base)[0]
        else:
            idxs_tbc = np.array(idxs_tbc)
            idxs_tbc = idxs_tbc[~self.__fv_in_base[idxs_tbc]]
        self.__products_fv_B1l[idxs_tbc] = self.__feature_vectors[idxs_tbc] @ self.__B1[:,self.__l]
        idxs_tbc = idxs_tbc[[not EQUAL_ZERO(p) for p in self.__products_fv_B1l[idxs_tbc]]]
        if not self.__hold_direction:
            self.__products_fv_B1l[idxs_tbc] = -self.__products_fv_B1l[idxs_tbc]
        idxs_tbc = idxs_tbc[[not (pos ^ (p>0)) for pos,p in zip(self.__fv_on_positive_site[idxs_tbc], self.__products_fv_B1l[idxs_tbc])]]
        if not spread_edges:
            distances = (1. - self.__products_fv_vertex[idxs_tbc]) / self.__products_fv_B1l[idxs_tbc]
        else:
            distances = (2. + idxs_tbc - self.__products_fv_vertex[idxs_tbc]) / self.__products_fv_B1l[idxs_tbc]
        return [(True, i, dist) for i,dist in zip(idxs_tbc, distances)]
    

    def __unit_vectors_on_exit_edge(self, idxs_tbc=None, spread_edges=False):
        """
        Parameters
        ----------
        idxs_tbc : Iterable[int], default None
            The list of indices of feature vectors to be considered as candidates to appear on the output edge
        spread_edges : bool, default False
            Whether to spread the edges crossing the exit edge, needed for degeneration

        Returns
        -------
        uv_seq : list[tuple[bool,int,float]]
            Information about unit vectors found on exit edge.
        """
        if idxs_tbc is None:
            idxs_tbc = np.nonzero(~self.__uv_in_base)[0]
        else:
            idxs_tbc = np.array(idxs_tbc)
            idxs_tbc = idxs_tbc[~self.__uv_in_base[idxs_tbc]]
        products_uv_B1l = self.__B1[idxs_tbc,self.__l]
        filter = [not EQUAL_ZERO(p) for p in products_uv_B1l]
        idxs_tbc, products_uv_B1l = idxs_tbc[filter], products_uv_B1l[filter]
        if not self.__hold_direction:
            products_uv_B1l = -products_uv_B1l
        filter = [not ((ev>0) ^ (p<0)) for ev,p in zip(self.__ev[idxs_tbc], products_uv_B1l)]
        idxs_tbc, products_uv_B1l = idxs_tbc[filter], products_uv_B1l[filter]
        if not spread_edges:
            distances = -self.__vertex[idxs_tbc] / products_uv_B1l
        else:
            filter = [not ((self.__dim + 1. + i - v > 0) ^ (p>0)) for i,v,p in zip(idxs_tbc, self.__vertex[idxs_tbc], products_uv_B1l)]
            idxs_tbc, products_uv_B1l = idxs_tbc[filter], products_uv_B1l[filter]
            distances = (self.__dim + 1. + idxs_tbc - self.__vertex[idxs_tbc]) / products_uv_B1l
        return [(False, i, dist) for i,dist in zip(idxs_tbc, distances)]
    

    def __go_to_new_vertex(self) -> bool:
        """
        Move along the edge of the exit until the gradient is greater than zero.

        Returns
        -------
        state : bool
            True - The gradient has reset to zero.
            False - The gradient is greater than zero and there is nowhere else to go.
        """
        # sort sequence on exit edge
        self.__seq.sort(key=lambda x: x[2])

        self.__kv = None
        for i, v in enumerate(self.__seq):
            self.__kv = (*v, i)
            if v[0]:
                self.__s -= abs(self.__products_fv_B1l[v[1]]) * self.__sample_weight[v[1]]
                if self.__products_fv_B1l[v[1]] > 0:
                    self.__modify_cv_plus(v[1])
                elif (self.__s > 0) and (i+1 < len(self.__seq)):
                    self.__modify_cv_minus(v[1])
                if self.__s <= 0:
                    break
            else:
                self.__s -= 2 * self.__p_lambda * abs(self.__B1[v[1],self.__l]) * self.__feature_weight[v[1]]
                if self.__s <= 0:
                    break
                self.__modify_cv_unit(v[1])

        return self.__s <= 0
        
    
    def __modify_cv_minus(self, fv_id):
        self.__fv_on_positive_site[fv_id] = True
        self.__cv += self.__feature_vectors[fv_id] * self.__sample_weight[fv_id]


    def __modify_cv_plus(self, fv_id):
        self.__fv_on_positive_site[fv_id] = False
        self.__cv -= self.__feature_vectors[fv_id] * self.__sample_weight[fv_id]


    def __modify_cv_unit(self, uv_id):
        self.__cv[uv_id] += 2 * self.__p_lambda * self.__ev[uv_id] * self.__feature_weight[uv_id]
        self.__ev[uv_id] *= -1


    def __update_to_new_vertex(self) -> None:
        """
        Change the vector in the base, update the optimisation process parameters.
        """
        B1_l = self.__B1[:,self.__l].copy()

        # base vector exchange
        self.__change_base()

        # if self.procedure_state in (ProcedureState.DEGENERATION_STATE, ProcedureState.BEGIN_DEGENERATION_STATE):
        #     # remove the vector kv entering the base from the degeneration sequence
        #     self.epsilon_neighbourhood.remove((self.kv[0], self.kv[1]))
        #     # add the vector lv outgoing from the base from the degeneration sequence
        #     if self.procedure_state == ProcedureState.DEGENERATION_STATE:
        #         self.epsilon_neighbourhood.add((self.lv[0], self.lv[1]))

        # coordinates of the new vertex
        self.__vertex[self.__uv_in_base] = 0.
        correction = self.__kv[2] * B1_l[~self.__uv_in_base]
        if self.__hold_direction:
            self.__vertex[~self.__uv_in_base] += correction
        else:
            self.__vertex[~self.__uv_in_base] -= correction

        # update self.__products_fv_vertex
        if self.__procedure_state == ProcedureState.NORMAL_STATE:
            self.__products_fv_vertex[self.__fv_in_base] = 1.0
            self.__products_fv_vertex[~self.__fv_in_base] += self.__kv[2] * self.__products_fv_B1l[~self.__fv_in_base]
        # else:
        #     idxs_tbc = [v[1] for v in self.epsilon_neighbourhood if v[0]]
        #     self.fvs.update_products_fv_vertex(self.kv, idxs_tbc)
        #     if self.lv[0]:
        #         self.fvs.recalculate_product_fv_vertex(self.lv[1], self.vertex, self.fs)

        # if self.procedure_state == ProcedureState.BEGIN_DEGENERATION_STATE:
        #     self.procedure_state = ProcedureState.DEGENERATION_STATE


    def __change_base(self):
        """
        Change the vector in the base B, update the reverse base B1.
        """
        # 1. update B1
        dots_kv_B1 = self.__feature_vectors[self.__kv[1]] @ self.__B1 if self.__kv[0] else self.__B1[self.__kv[1]] * self.__ev[self.__kv[1]]
        col_l = self.__B1[:,self.__l] / dots_kv_B1[self.__l]
        
        # modification of the other columns B1
        self.__B1 -= dots_kv_B1 * np.array([col_l]).T
        # modification of column B1[:,l]
        self.__B1[:,self.__l] = col_l

        # 2. update B
        if self.__lv[0]:
            self.__fv_in_base[self.__lv[1]] = False
        else:
            self.__uv_in_base[self.__lv[1]] = False
        self.__B_type[self.__l], self.__B_index[self.__l] = self.__kv[:2]
        if self.__kv[0]:
            self.__fv_in_base[self.__kv[1]] = True
        else:
            self.__uv_in_base[self.__kv[1]] = True


    def __check_degeneration(self):
        # try to create epsilon neighbourhood sequence
        degeneration_detected = self.__find_epsilon_neighbourhood()
        if not degeneration_detected:
            return False


    def __find_epsilon_neighbourhood(self) -> bool:
        """help with checking degeneration"""
        kv_seq_id = self.__kv[3]
        self.__epsilon_neighbourhood = set()

        if (kv_seq_id > 0) and EQUALS_EPSILON(self.__kv[2], self.__seq[kv_seq_id-1][2]):
            i = kv_seq_id-1
            while (i >= 0) and EQUALS_EPSILON(self.__kv[2], self.__seq[i][2]):
                self.__epsilon_neighbourhood.add(self.__seq[i][:2])
                i -= 1

        if (kv_seq_id < len(self.__seq)-1) and (EQUALS_EPSILON(self.__kv[2], self.__seq[kv_seq_id+1][2])):
            new_kv = self.__seq[kv_seq_id]
            i = kv_seq_id+1
            while (i < len(self.__seq)) and EQUALS_EPSILON(self.__kv[2], self.__seq[i][2]):
                self.__epsilon_neighbourhood.add(new_kv[:2])
                if new_kv[0]:
                    if self.__products_fv_B1l[new_kv[1]] <= 0:
                        self.__modify_cv_minus(new_kv[1])
                else:
                    self.__modify_cv_unit(new_kv[1])
                new_kv = self.__seq[i]
                if new_kv[0]:
                    if self.__products_fv_B1l[new_kv[1]] > 0:
                        self.__modify_cv_plus(new_kv[1])
                    self.__s -= abs(self.__products_fv_B1l[new_kv[1]]) * self.__sample_weight[new_kv[1]]
                else:
                    self.__s -= 2 * self.__p_lambda * abs(self.__B1[new_kv[1],self.__l]) * self.__feature_weight[new_kv[1]]
                    self.__modify_cv_unit(new_kv[1])
                i += 1
            self.__kv = new_kv

        return len(self.__epsilon_neighbourhood) > 0


    def _end_degeneration(self):
        """ending of degeneration and return to normal state"""        
        pass


    def __optimal_vertex(self):
        """determination of the base vertex not moved and products of <fv,v> in the non-moving vertex"""
        pass
        # self.vertex = self.bases.get_current_vertex()
        # self.fvs.recalculate_products_fv_vertex(self.vertex, self.fs)


    def __get_current_vertex(self):
        """
        Determine, without iterative calculations, the coordinates of the current vertex
        where the opimisation procedure is located.
        """
        vertex = self.__B1[:,self.__B_type].sum(axis=1)
        return vertex
    

    def __func_crit(self, vertex):
        pyvs = self.__feature_vectors @ vertex
        fv_on_positive_side = pyvs+1e-10 < 1
        cr1 = ((1.-pyvs[fv_on_positive_side]) * self.__sample_weight[fv_on_positive_side]).sum()
        cr2 = self.__p_lambda * np.absolute(vertex * self.__feature_weight).sum()
        return cr1, cr2
    

    def __margin_width(self, vertex):
        dem = np.absolute(vertex[~self.__uv_in_base]*self.__feature_weight[~self.__uv_in_base]).sum()
        return 1 / dem if dem > 0 else INFINITY
    

    def __correction_vector(self, vertex):
        pyvs = self.__feature_vectors @ vertex
        sv_on_positive_side = pyvs+1e-10 < 1
        cv = (self.__feature_vectors[sv_on_positive_side] * np.array([self.__sample_weight[sv_on_positive_side]]).T).sum(axis=0)
        cv -= self.__p_lambda * (1. - 2*(vertex < 0)) * self.__feature_weight
        return cv


    def __print_log(self, verbose: int, n_step: int = None):
        def v_info(v):
            if v is None:
                return "None"
            if v[0]:
                return f"{v[1]}d"
            else:
                return f"{v[1]}{('-','+')[self.__ev[v[1]]==1]}e"
        if verbose >= 1:
            print(f"----{n_step}----------")
            #print("Diagnose info")
            vertex = self.__get_current_vertex()
            #print("vertex: ", sorted([(f,v) for f,v in zip(self.__features[~self.__uv_in_base],vertex[~self.__uv_in_base])], key=lambda x:x[0]))
            print(f"Fc = {self.__func_crit(vertex)}")
            #print(f"Margin = {self.__margin_width(vertex)}")
            #print(f"procedure state = {self.__procedure_state}")
            #if self.__procedure_state == ProcedureState.DEGENERATION_STATE:
            #    print()
            #    print(self.__epsilon_neighbourhood)
            #print()
            print(f"l={self.__l}, lv={v_info(self.__lv)}, kv={v_info(self.__kv)},{self.__kv[2:] if self.__kv is not None else ''}")
            print()
            print("B:")
            for v in zip(self.__B_type, self.__B_index):
                print(f"{'_'.join([str(v[0]),str(v[1])]):>10s}", end=" | ")
                if v[0]:
                    print("".join([f"{val:>7.2f}" for val in self.__feature_vectors[v[1]]]))
                else:
                    print("".join([f"{0.0 if f1!=v[1] else self.__ev[v[1]]:>7.2f}" for f1 in range(self.__dim)]))
            print()
            print("B1")
            for row in self.__B1:
                print("".join([f"{val:>7.2f}" for val in row]))
            #print()
            #print(f"seq: {self.__seq}")
            #print()
            #print("cv(proc): ", self.__cv)
            #print("cv(spr ): ", self.__correction_vector(vertex))
            # for cv_p, cv_s in zip(self.__cv[:10],self.__correction_vector(vertex)[:10]):
            #     if not EQUALS_EPSILON(cv_p, cv_s):
            #         print("--->", cv_p, cv_s) 
            # print()
            # print("B1 x cv: ", self.__cv @ self.__B1)
            # print()
            # print("vertex(spr ): ", sorted([(f,v) for f,v in zip(self.__features[~self.__uv_in_base],vertex[~self.__uv_in_base])], key=lambda x:x[0]))
            # print("vertex(proc): ", sorted([(f,v) for f,v in zip(self.__features[~self.__uv_in_base],self.__vertex[~self.__uv_in_base])], key=lambda x:x[0]))
            # print()
            # print("# proc_ps man_ps man_pyv error")
            # pyvs = self.__feature_vectors @ vertex
            # for i,(pyv,ps) in enumerate(zip(pyvs, self.__fv_on_positive_site)):
            #     print(i, ps, pyv+1e-10 < 1, pyv, "<---" if ps != (pyv+1e-10 < 1) else "")
            #print("--------------")


    def print_state_info(self):
        """
        Print information about the current status of the calculation procedure.
        """
        vertex = self.__get_current_vertex()
        important_features = sorted(self.__features[~self.__uv_in_base])
        fc = self.__func_crit(vertex)

        # basic parameters
        print("------------------------")
        print("Basic parameters:")
        print("Fc = ", f"{fc[0]:.6f} + {fc[1]:.10f}")
        print("Margin = ", f"{self.__margin_width(vertex):.6f}")
        print(f"Features: #{len(self.__features)} {sorted(self.__features)}")
        print(f"Important features: #{len(important_features)} {important_features}")

        # vertex
        print("------------------------")
        print("Vertex:")
        print("v = ", [(self.__features[f],round(vertex[f],6)) for f in np.nonzero(~self.__uv_in_base)[0]])

        print("------------------------")
