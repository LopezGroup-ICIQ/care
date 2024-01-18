import pandas as pd

import care.crn.utilities.additional_funcs as af


class Printer:
    def __init__(self, network, vader=None):
        self.network = network
        self.vader = vader

    def get_df_inter(self):
        header = {
            "label": [],
            "formula": [],
            "phase": [],
            "G": [],
            "mw": [],
            "ne": [],
            "frq": [],
        }

        for comps in [self.network.intermediates, self.network.gasses]:
            for label, inter in comps.items():
                elems = inter.molecule.elements_number
                if inter.is_surface:
                    weigth = 0
                else:
                    weigth = af.calculate_weigth(elems)
                if inter.phase != "cat":
                    init_label = "g"
                else:
                    init_label = "i"
                header["label"].append(init_label + label)
                header["formula"].append(af.code_mol_graph(inter.graph, elems=["C"]))
                header["phase"].append(inter.phase)
                header["G"].append(inter.energy)
                header["mw"].append(weigth)
                header["ne"].append(inter.electrons)
                header["frq"].append("[]")

            if self.vader:
                if "nv" not in header:
                    header["nv"] = []
                for label in comps:
                    if label in self.vader:
                        vad_val = "{: 5.2f}".format(self.vader[label])
                    else:
                        vad_val = "unknwn"
                    header["nv"].append(vad_val)
        df_out = pd.DataFrame(data=header)
        if self.vader:
            new_col = list(df_out.columns)
            new_col[-1], new_col[-2] = new_col[-2], new_col[-1]
            df_out = df_out.reindex(columns=new_col)
        return df_out

    def get_df_ts(self):
        header = {
            "label": [],
            "is1": [],
            "is2": [],
            "fs1": [],
            "fs2": [],
            "G": [],
            "alpha": [],
            "beta": [],
            "frq": [],
        }

        for t_state in self.network.t_states:
            order = t_state.full_order()
            order_pvt = t_state.full_order()

            for item in order:
                for index, inter in enumerate(item):
                    if inter.phase == "cat":
                        item[index] = "i" + inter.code
                    elif inter.code in "e-":
                        item[index] = "None"
                    else:
                        item[index] = "g" + inter.code
            if t_state.is_electro:
                begin = "aqu"
                react_ener = sum([inter_pvt.energy for inter_pvt in order_pvt[0]])
                prod_ener = sum([inter_pvt.energy for inter_pvt in order_pvt[1]])
                if react_ener > prod_ener:
                    alpha = 0
                else:
                    alpha = 1
                beta = 0.05
            elif t_state.r_type == "ads":
                begin = "ads"
                alpha = 0
                beta = 0
            else:
                begin = "rxn"
                alpha = af.INTERPOL[t_state.r_type]["alpha"]
                beta = af.INTERPOL[t_state.r_type]["beta"]
            is1, is2 = order[0]
            fs1, fs2 = order[1]
            header["is1"].append(is1)
            header["is2"].append(is2)
            header["fs1"].append(fs1)
            header["fs2"].append(fs2)
            for inter_code in (is1, is2, fs1, fs2):
                if inter_code is None:
                    begin += "xxxxxx"
                begin += inter_code
            header["label"].append(begin)
            header["G"].append(t_state.energy)
            header["alpha"].append(alpha)
            header["beta"].append(beta)
            header["frq"].append("[]")

        return pd.DataFrame(data=header)

    def write(self, filename, header, dataf, header_fmt, body_fmt):
        with open(filename, "w") as out_file:
            out_file.write(header_fmt.format(*header))
            for _, item in dataf.iterrows():
                line = list(item)
                out_file.write(body_fmt.format(*line))

    def write_inter(self, filename):
        inter_df = self.get_df_inter()
        header = list(inter_df.columns)
        header_fmt = "{:^7} {:^35} {:^7} {:^12} {:^6} {:^2} {:^20}\n"
        inter_fmt = "{:7} {:35} {:7} {: 12.8f} {: 6.2f} {:2} {:20}\n"
        if "nv" in header:
            header_tmp = header_fmt.split(" ")
            header_tmp.insert(6, "{}")
            header_fmt = " ".join(header_tmp)
            inter_tmp = inter_fmt.split(" ")
            inter_tmp.insert(6, "{}")
            inter_fmt = " ".join(inter_tmp)
        self.write(filename, header, inter_df, header_fmt, inter_fmt)

    def write_ts(self, filename):
        ts_df = self.get_df_ts()
        header = list(ts_df.columns)
        header_fmt = "{:^31} {:^7} {:^7} {:^7} {:^7} {:^12} {:^5} {:^5} {:^20}\n"
        ts_fmt = "{:31} {:7} {:7} {:7} {:7} {: .8f} {:^5.2f} {:^5.2f} {:^20}\n"
        self.write(filename, header, ts_df, header_fmt, ts_fmt)
