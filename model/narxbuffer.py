import torch
import pandas as pd

class NarxBuffer:
    def __init__(
        self,
        endog_variable_names,
        exog_variable_names,
        model_in_variable_order,
        model_out_variable_order,
        t_endog,
        t_exog,
    ):
        self._buffer = pd.DataFrame(
            columns=list(set(endog_variable_names).union(exog_variable_names)),
            index=[f"t - {i}" for i in range(1, max(t_endog, t_exog) + 1)],
        )
        self._t_endog = t_endog
        self._t_exog = t_exog
        self._endog_variable_names = endog_variable_names
        self._exog_variable_names = exog_variable_names
        self._model_in_variable_order = model_in_variable_order
        self._model_out_variable_order = model_out_variable_order

    def populate_buffer_from_xlsx(self, xlsx_path: str):
        df = pd.read_excel(xlsx_path).drop(["Date"], axis=1)
        self.populate_buffer_from_df(df)

    def populate_buffer_from_df(self, df: pd.DataFrame):
        assert all(col in self._buffer.columns for col in df.columns) and all(
            col in df.columns for col in self._buffer.columns
        )
        df = df.sort_index(ascending=False)
        for col in self._endog_variable_names:
            self._buffer[col] = df[col][: self._t_endog].values
        for col in self._exog_variable_names:
            self._buffer[col] = df[col][: self._t_exog].values

    def feed_model(self, c, p, l):
        model_input_vars = {}
        for var in self._model_in_variable_order:
            if var == "C":
                model_input_vars["C"] = c
            elif var == "P":
                model_input_vars["P"] = p
            elif var == "L":
                model_input_vars["L"] = l
            else:
                # Lags
                var_base_name = var[: var.find("(")]
                lag = int(var.split("(")[-1].split("-")[-1].replace(")", ""))
                model_input_vars[var] = self._buffer[var_base_name][f"t - {lag}"]

        # PUSH C P L IN BUFFER
        self._buffer = self._buffer.shift(1)
        self._buffer.loc["t - 1", "C"] = c
        self._buffer.loc["t - 1", "P"] = p
        self._buffer.loc["t - 1", "L"] = l
        return torch.Tensor(list(model_input_vars.values()))

    def update_buffer(self, model_out):
        for var, var_name in zip(model_out, self._model_out_variable_order):
            self._buffer.loc["t - 1", var_name] = var.item()