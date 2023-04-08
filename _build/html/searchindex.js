Search.setIndex({"docnames": ["book/intro", "notebooks/T2_linear_programming/01_linear_programming", "notebooks/T2_linear_programming/02_transportation_problem", "notebooks/T2_linear_programming/03a_Simplex_Method", "notebooks/T2_linear_programming/03b_Simplex_Method", "notebooks/T2_linear_programming/04_String_beam_system", "notebooks/T3_mixed_integer_linear_programming/01_Knapsack_Problem", "notebooks/T3_mixed_integer_linear_programming/02_The_travelling_salesman_problem", "notebooks/T4_nonlinear_programming/01_Sand_Transport", "notebooks/T4_nonlinear_programming/02_Postal_package", "notebooks/T4_nonlinear_programming/03_String_beam_system_Linear", "notebooks/T4_nonlinear_programming/03_String_beam_system_NonLinear", "notebooks/T5_heuristic_optimization/SCE_Scipy_Example", "notebooks/T6_metamodels/01a_Selection_MDA", "notebooks/T6_metamodels/01b_LHS_MDA", "notebooks/T6_metamodels/01c_Clustering_KMeans", "notebooks/T6_metamodels/02a_Reconstruction_Analogues", "notebooks/T6_metamodels/02b_Reconstruction_RBF"], "filenames": ["book/intro.md", "notebooks/T2_linear_programming/01_linear_programming.ipynb", "notebooks/T2_linear_programming/02_transportation_problem.ipynb", "notebooks/T2_linear_programming/03a_Simplex_Method.ipynb", "notebooks/T2_linear_programming/03b_Simplex_Method.ipynb", "notebooks/T2_linear_programming/04_String_beam_system.ipynb", "notebooks/T3_mixed_integer_linear_programming/01_Knapsack_Problem.ipynb", "notebooks/T3_mixed_integer_linear_programming/02_The_travelling_salesman_problem.ipynb", "notebooks/T4_nonlinear_programming/01_Sand_Transport.ipynb", "notebooks/T4_nonlinear_programming/02_Postal_package.ipynb", "notebooks/T4_nonlinear_programming/03_String_beam_system_Linear.ipynb", "notebooks/T4_nonlinear_programming/03_String_beam_system_NonLinear.ipynb", "notebooks/T5_heuristic_optimization/SCE_Scipy_Example.ipynb", "notebooks/T6_metamodels/01a_Selection_MDA.ipynb", "notebooks/T6_metamodels/01b_LHS_MDA.ipynb", "notebooks/T6_metamodels/01c_Clustering_KMeans.ipynb", "notebooks/T6_metamodels/02a_Reconstruction_Analogues.ipynb", "notebooks/T6_metamodels/02b_Reconstruction_RBF.ipynb"], "titles": ["M2248 - Optimization in Civil Engineering", "Linear Programming", "Transportation problem", "SIMPLEX Method - Ex1", "SIMPLEX Method - Ex2", "String and beam system", "The Knapsack problem", "The Travelling Salesman", "Sand Transport", "Postal Package", "String and beam system - Linear", "String and beam system - NonLinear", "Heuristic optimization", "Selection: MDA", "Sampling LHS + Selection MDA", "Clustering - KMeans", "Reconstruction - Analogues", "Reconstruction: RBF"], "terms": {"fernando": 0, "j": [0, 2, 7], "m\u00e9ndez": 0, "incera": 0, "mendez": 0, "unican": 0, "es": [0, 12], "laura": 0, "cagig": 0, "gil": 0, "paula": 0, "camu": 0, "import": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], "warn": [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16], "filterwarn": [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16], "ignor": [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16], "os": [1, 2, 13, 14, 15], "path": [1, 2, 13, 15], "op": [1, 2, 13, 15], "matplotlib": [1, 2, 5, 6, 10, 11, 12, 13, 15, 16, 17], "pyplot": [1, 2, 5, 6, 10, 11, 12, 13, 15, 16, 17], "plt": [1, 2, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17], "numpi": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], "np": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], "from": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], "scipi": [1, 2, 5, 8, 9, 10, 11, 13, 14, 15, 17], "optim": [1, 2, 5, 6, 7, 8, 9, 10, 11], "linprog": [1, 2, 5, 10], "we": [1, 2, 5, 6, 7, 9, 10, 11, 12, 16, 17], "ar": [1, 2, 3, 5, 6, 7, 8, 10, 11, 13, 17], "go": [1, 2, 6, 7, 12, 17], "us": [1, 2, 3, 4, 6, 7, 13, 15], "funcion": [1, 2, 12], "librari": [1, 2, 6, 7, 13], "The": [1, 2, 3, 4, 5, 8, 9, 10, 11, 12], "constraint": [1, 2, 6, 7, 9, 10, 11, 12], "thi": [1, 2, 3, 4, 5, 6, 7, 12, 13, 15, 16], "can": [1, 2, 5, 6, 7, 9, 10, 11, 17], "onlin": [1, 2], "minim": [1, 2, 3, 5, 7, 8, 9, 10, 11, 12], "function": [1, 2, 5, 6, 7, 8, 9, 10, 11, 15, 17], "inequ": [1, 2, 5, 9, 10, 11], "need": [1, 2, 5, 9, 10, 11, 16], "given": [1, 2, 5, 6, 7, 9, 10, 11], "leq": [1, 2, 3, 4, 5, 6, 7, 10, 11, 12], "maxim": [1, 3, 4, 5, 6, 9, 10, 11], "z": [1, 2, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15, 16, 17], "x": [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], "2y": 1, "subject": [1, 3, 4, 6, 7], "2x": 1, "y": [1, 2, 7, 8, 9, 13, 14, 15, 16, 17], "20": [1, 12, 13, 15, 16, 17], "4x": 1, "5y": 1, "10": [1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 17], "geq": [1, 2, 3, 4, 5, 10, 11, 12], "2": [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], "15": [1, 2, 4, 6, 9, 12, 13, 14, 15, 16, 17], "0": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], "As": [1, 5, 9, 10, 11], "first": [1, 5, 6, 7, 10, 11, 16, 17], "transform": [1, 3, 4, 16], "so": [1, 5, 10, 11], "problem": [1, 3, 4, 7, 8, 9, 10, 11], "consist": 1, "translat": 1, "sinc": 1, "doesn": 1, "t": [1, 12, 13, 14, 15, 16, 17], "recogn": 1, "step": [1, 14], "defin": [1, 2, 5, 6, 7, 8, 9, 10, 11, 12], "c": [1, 2, 5, 6, 7, 10, 11, 13, 14, 15, 16, 17], "a_ub": 1, "none": 1, "b_ub": 1, "a_eq": [1, 2, 5], "b_eq": [1, 2, 5], "bound": [1, 2, 5, 8, 9, 10, 11, 12], "where": [1, 2, 3, 4, 6], "coeffici": [1, 2], "object": [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 16], "matrix": [1, 7, 13, 16, 17], "left": [1, 2, 5, 10], "side": [1, 2, 5, 8, 10], "vector": [1, 13, 16, 17], "right": [1, 2, 5, 10], "equal": [1, 2, 5, 10], "A": [1, 2, 5, 9, 10, 11], "sequenc": 1, "min": [1, 12, 13, 16, 17], "max": [1, 12, 13, 16, 17], "pair": [1, 7], "each": [1, 2, 3, 4, 6, 7, 8, 9, 16], "element": [1, 3, 4, 6, 7], "valu": [1, 3, 4, 6, 7, 12, 14, 16, 17], "all": [1, 3, 4, 11, 17], "variabl": [1, 6, 7, 13, 14, 15], "obj": [1, 2, 5], "1": [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], "lhs_ineq": 1, "4": [1, 2, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17], "5": [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], "rhs_ineq": 1, "lhs_eq": [1, 2, 5, 10], "rhs_eq": [1, 2, 5, 10], "bnd": [1, 2, 5, 8, 9, 10, 11], "inf": [1, 2, 5, 8, 9, 10, 11], "plot": [1, 2, 6, 12, 13, 16, 17], "feasibl": 1, "region": 1, "min_x": 1, "max_x": 1, "assess": 1, "min_i": 1, "max_i": 1, "linspac": [1, 12, 13, 16, 17], "150": [1, 2, 6, 17], "xx": [1, 12], "yy": [1, 12], "meshgrid": [1, 12, 13, 16, 17], "100": [1, 5, 6, 8, 10, 11, 12, 15, 16, 17], "creat": [1, 7, 16, 17], "mask": 1, "index": [1, 7, 13, 14, 15], "full": [1, 2], "shape": [1, 7, 13, 15, 16, 17], "ie": 1, "rang": [1, 2, 6, 7, 13, 14, 15], "len": [1, 6, 7, 13, 14, 15, 17], "nan": 1, "zz": [1, 12], "fig": [1, 2, 6, 12, 13, 14, 15, 16, 17], "ax": [1, 2, 6, 8, 9, 12, 13, 14, 15, 16, 17], "subplot": [1, 2, 6, 12, 13, 17], "figsiz": [1, 2, 6, 9, 12, 13, 14, 15, 16, 17], "9": [1, 2, 7, 12, 13, 14, 15, 17], "6": [1, 2, 5, 6, 7, 9, 10, 11, 13, 14, 15], "set_xlabel": [1, 6, 8, 13, 14, 15, 16, 17], "fontsiz": [1, 2, 6, 12, 13, 14, 15, 16, 17], "set_ylabel": [1, 6, 8, 13, 14, 15, 16, 17], "size": [1, 13, 14, 15, 17], "label": [1, 2, 6, 15, 16], "str": [1, 2, 11], "linewidth": [1, 6, 12, 13, 14, 15], "3": [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], "zorder": [1, 2, 13, 14, 15, 16, 17], "im": [1, 6, 13, 14, 15, 16, 17], "pcolormesh": 1, "cmap": [1, 6, 12, 13, 14, 15, 16, 17], "rainbow": [1, 6, 12, 13, 14, 15, 16, 17], "im2": 1, "contour": 1, "color": [1, 2, 6, 7, 12, 13, 14, 15, 16, 17], "black": [1, 15, 16, 17], "linestyl": [1, 2], "legend": [1, 6, 12, 16], "set_xlim": [1, 16], "set_ylim": [1, 16], "colorbar": [1, 6, 12, 13, 15, 16, 17], "0x7f9ca4e59d60": 1, "appli": [1, 2, 5, 10, 16, 17], "opt": [1, 2, 5], "messag": [1, 2, 5, 8, 9, 10, 11], "termin": [1, 2, 5, 8, 9, 10, 11], "successfulli": [1, 2, 5, 8, 9, 10, 11], "high": [1, 2, 5], "statu": [1, 2, 5, 8, 9, 10, 11], "7": [1, 2, 3, 4, 5, 12, 13, 14, 15, 16], "success": [1, 2, 5, 8, 9, 10, 11], "true": [1, 2, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16], "fun": [1, 2, 5, 8, 9, 10, 11, 12], "16": [1, 3, 4, 7, 12, 13, 14, 17], "818181818181817": 1, "727e": 1, "00": [1, 2, 5, 6, 7, 8, 10, 11, 15], "545e": 1, "nit": [1, 2, 5, 8, 9, 10, 11], "lower": [1, 2, 5], "residu": [1, 2, 5], "margin": [1, 2, 5], "000e": [1, 2, 5, 10, 11], "upper": [1, 2, 5], "eqlin": [1, 2, 5], "01": [1, 2, 5, 7, 8, 9, 15], "ineqlin": [1, 2, 5], "818e": 1, "364e": 1, "mip_node_count": [1, 2, 5], "mip_dual_bound": [1, 2, 5], "mip_gap": [1, 2, 5], "result": [1, 2, 6, 7, 8, 9, 10, 11, 17], "k": [1, 8, 13, 15, 16], "markers": 1, "ncol": 1, "grid": [1, 13, 14, 15], "alpha": [1, 13, 14, 15, 16, 17], "0x7f9ca5d4ae50": 1, "If": 1, "remov": 1, "follow": [1, 2, 6, 7, 9], "opt_no_eq": 1, "method": [1, 8, 9, 11, 12, 16], "714285714285715": 1, "429e": 1, "143e": 1, "857e": 1, "286e": 1, "0x7f9ca68faf70": 1, "unbound": 1, "model_statu": 1, "primal_statu": 1, "At": 1, "fals": [1, 12], "appear": 1, "try": [1, 12, 15, 16], "except": [1, 7, 12], "print": [1, 3, 4, 5, 6, 9, 10, 11, 12, 17], "No": 1, "solut": [1, 3, 4, 5, 6, 7, 9, 10], "found": [1, 3, 4, 7], "0x7f9ca74c68e0": 1, "chang": [1, 7], "5x": 1, "0x7f9ca7ffbdc0": 1, "easili": 1, "see": 1, "graphic": 1, "how": 1, "infinit": 1, "number": [1, 2, 7, 8, 13, 14, 15, 16], "ly": 1, "over": [1, 5, 8, 10, 11, 17], "line": [1, 6, 7, 13], "nevertheless": 1, "onli": [1, 5, 7, 9, 10, 11], "give": [1, 6, 7], "one": [1, 7, 8], "those": 1, "assum": [1, 5, 8, 10, 11], "prefer": 1, "ani": 1, "should": [1, 3, 4], "includ": [1, 7], "more": [1, 5, 10, 11], "exampl": [1, 6, 7, 16], "wont": 1, "elif": 1, "axhlin": 1, "firebrick": [1, 2, 6, 12], "els": [1, 7, 12], "axvlin": [1, 6], "0x7f9ca8b21700": 1, "labellin": 2, "good": 2, "sourc": 2, "warehous": 2, "variou": [2, 6], "destin": 2, "locat": [2, 7, 11], "minimum": [2, 8], "cost": [2, 8], "There": 2, "differ": [2, 6, 7, 16], "m": [2, 6], "three": [2, 8], "distin": 2, "n": [2, 6, 7, 8, 9, 12, 14, 16], "own": 2, "product": [2, 7], "demand": 2, "respect": [2, 8], "m1": 2, "m2": 2, "m3": 2, "300": [2, 5, 10, 11], "600": 2, "n1": 2, "n2": 2, "n3": 2, "450": 2, "900": 2, "8": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], "xm": 2, "ym": 2, "xn": 2, "yn": 2, "scatter": [2, 6, 13, 14, 15, 16, 17], "marker": [2, 7, 16], "s": [2, 6, 7, 12, 13, 14, 15, 16, 17], "800": [2, 5, 10, 11], "navi": 2, "3500": 2, "mi": 2, "text": [2, 6, 7, 8, 12, 13, 16, 17], "white": 2, "verticalalign": 2, "center": [2, 7, 15], "horizontalalign": 2, "fontweight": 2, "bold": 2, "ni": [2, 12], "dash": 2, "x_": [2, 7], "axi": [2, 12, 13, 15, 16, 17], "off": 2, "gca": 2, "get_lin": 2, "17": [2, 6, 12, 13, 15], "ha": [2, 5, 6, 9, 10, 11, 13], "total": [2, 5, 6, 7, 9, 10, 11], "ship": 2, "sum_": [2, 6, 7], "i": [2, 6, 7, 12, 16], "c_": 2, "ij": 2, "amount": 2, "associ": 2, "per": [2, 8], "unit": 2, "structur": [2, 5, 9, 10, 11], "11": [2, 4, 7, 15], "12": [2, 4, 6, 12, 15], "13": [2, 3, 13, 15], "21": [2, 9, 14, 15], "22": 2, "23": 2, "31": [2, 12], "32": 2, "33": 2, "suppli": 2, "b": [2, 3, 4, 5, 10, 11, 12], "arriv": 2, "goo": 2, "posit": [2, 5, 7, 10, 11], "4800": 2, "02": [2, 5, 9, 10, 11, 15], "500e": 2, "100e": 2, "cont": 2, "panda": [3, 4, 13, 14, 15, 16], "pd": [3, 4, 13, 14, 15, 16], "simplex_method_geocean": [3, 4], "get_new_tableau": [3, 4], "check_optim": [3, 4], "get_optimal_result": [3, 4], "an": [3, 4, 7, 8, 9, 11], "approach": [3, 4], "solv": [3, 4, 5, 6, 10, 11], "linear": [3, 4, 6, 7], "program": [3, 4, 6, 7], "hand": [3, 4], "mean": [3, 4, 12], "find": [3, 4, 7, 8], "8x_1": 3, "10x_2": 3, "7x_3": 3, "x_1": [3, 4, 5, 10, 11, 12], "3x_2": 3, "2x_3": 3, "5x_2": [3, 4], "x_3": [3, 11, 12], "x_2": [3, 4, 5, 10, 11, 12], "turn": [3, 4], "construaint": [3, 4], "must": [3, 4, 9], "varibl": [3, 4], "s_1": [3, 4], "s_2": [3, 4], "column": [3, 4, 6, 7, 14, 15], "datafram": [3, 4, 14, 15, 16], "x1": [3, 4, 5, 10, 11, 12], "x2": [3, 4, 5, 10, 11, 12], "x3": [3, 11], "s1": [3, 4], "s2": [3, 4], "For": [3, 4, 6, 7, 9, 11], "last": [3, 4, 12], "row": [3, 4, 6, 7, 15], "NOT": [3, 4], "pivot_column": [3, 4], "argmin": [3, 4], "iloc": [3, 4], "select": [3, 4, 6, 16, 17], "kei": [3, 4, 15], "b_index": [3, 4], "divid": [3, 4], "b_mod": [3, 4], "row_piv_var": [3, 4], "format": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "333333": [3, 4, 6], "600000": 3, "dtype": [3, 4, 15], "float64": [3, 4, 14], "other": [3, 4, 5, 8, 10, 11], "comput": [3, 4], "tabl": [3, 4], "neg": [3, 4], "old": [3, 4], "new_tableau": [3, 4], "copi": [3, 4], "30": [3, 8, 10, 12, 13, 15, 16, 17], "64": [3, 14], "optimal_solut": [3, 4], "basic": [3, 4, 12, 15], "non": [3, 4], "4x_1": 4, "6x_2": 4, "27": 4, "2x_1": 4, "90": [4, 15], "s_3": 4, "s3": 4, "18": [4, 9, 12, 13, 15, 16, 17], "35": 4, "66": [4, 8], "285714": 4, "142857": 4, "428571": 4, "714286": 4, "116": 4, "14285714": [4, 15], "42857143": 4, "56": 4, "14": [4, 9, 12, 13, 14, 15], "666667": 4, "132": [4, 16], "66666667": 4, "pil": [5, 10, 11], "imag": [5, 10, 11], "below": [1, 2, 3, 4, 5, 10, 11], "load": [5, 7, 10, 11, 15, 16], "know": [5, 10, 11], "maximum": [5, 6, 10, 11], "d": [5, 10, 11, 12, 14], "200": [5, 10, 11, 15, 16, 17], "e": [5, 7, 10, 11, 13], "f": [5, 7, 10, 11, 12], "open": [5, 10, 11], "resourc": [5, 8, 9, 10, 11, 12], "sketch_viga": [5, 10], "png": [5, 10, 11], "resiz": [5, 10, 11], "550": [5, 10, 11], "lanczo": [5, 10, 11], "equilibrium": [5, 10, 11], "equat": [5, 10, 11], "t_e": [5, 10, 11], "t_f": [5, 10, 11], "t_c": [5, 10, 11], "t_d": [5, 10, 11], "t_a": [5, 10, 11], "t_b": [5, 10, 11], "Then": [5, 6, 10, 11], "obtain": [5, 6, 10, 11, 16, 17], "momentum": [5, 10, 11], "some": [5, 10, 11], "condit": [5, 10, 11], "restrict": [5, 10, 11], "than": [5, 10, 11], "reduc": [5, 10, 11, 17], "them": [5, 10, 11], "3000": [5, 10], "do": [5, 9, 10, 11], "640": 5, "400e": [5, 10, 11], "x0": [5, 8, 9, 10, 11, 12], "440": [5, 10], "mip": [6, 7], "model": [6, 7, 13, 16, 17], "xsum": [6, 7], "binari": [6, 7], "mix": 6, "integ": 6, "classic": 6, "involv": 6, "subset": [6, 13, 14, 15], "item": [6, 7], "highest": 6, "weight": [6, 7], "v_i": 6, "w_i": 6, "goal": 6, "while": [6, 7, 12], "ensur": [6, 7], "doe": 6, "exce": [6, 9], "capac": 6, "formul": [6, 7], "x_i": 6, "begin": [6, 7], "align": [6, 7], "dot": 6, "end": [6, 7], "decis": 6, "indic": [6, 7], "whether": 6, "solver": [6, 7], "satisfi": 6, "name": [6, 7, 13, 14, 15], "name_model": [6, 7], "v": [6, 7, 9], "w": 6, "set": [6, 7], "horizont": [6, 16], "crimson": [6, 17], "add": [6, 7], "titl": 6, "set_titl": [6, 8, 9, 12, 13, 16, 17], "displai": [6, 7], "set_label": [6, 13, 15, 16, 17], "id": [6, 7], "initi": [6, 7, 8, 9, 10, 11, 12], "iterm": [6, 7], "add_var": [6, 7], "var_typ": [6, 7], "welcom": [6, 7], "cbc": [6, 7], "milp": [6, 7], "version": [6, 7], "trunk": [6, 7], "build": [6, 7], "date": [6, 7, 15], "oct": [6, 7], "28": [6, 7, 12], "2021": [6, 7], "start": [6, 7, 14], "relax": [6, 7], "primal": [6, 7], "simplex": [6, 7], "coin0506i": [6, 7], "presolv": [6, 7], "clp1000i": [6, 7], "sum": [6, 7, 9, 16, 17], "infeas": [6, 7], "averag": [6, 7], "fix": [6, 7], "clp0000i": [6, 7], "coin0511i": [6, 7], "after": [6, 7], "postsolv": [6, 7], "dual": [6, 7], "clp0032i": [6, 7], "33333333": 6, "iter": [6, 7], "time": [6, 7, 8, 12], "002": 6, "idiot": [6, 7], "optimizationstatu": [6, 7], "extract": [6, 7, 14], "which": [6, 9], "input": 6, "inform": 6, "wether": 6, "been": 6, "dure": 6, "process": 6, "99": [6, 7], "objective_valu": [6, 7], "arrai": [6, 12, 13, 15, 17], "50": [6, 8, 11, 13, 15, 16], "lightgrei": 6, "Not": 6, "vmin": [6, 13, 15, 16, 17], "vmax": [6, 13, 15, 16, 17], "itertool": 7, "sy": [7, 14, 15], "stdout": 7, "out": 7, "folium": 7, "citi": 7, "ldot": 7, "distanc": [7, 16], "d_": 7, "repres": 7, "between": 7, "tsp": 7, "seek": 7, "shortest": 7, "hamiltonian": 7, "cycl": 7, "visit": 7, "exactli": 7, "onc": [7, 17], "specifi": 7, "g": 7, "let": 7, "take": 7, "tour": 7, "edg": 7, "otherwis": 7, "u_i": 7, "continu": 7, "foral": 7, "u_j": 7, "nx_": 7, "neq": 7, "second": 7, "third": 7, "known": 7, "subtour": 7, "elimin": 7, "prevent": 7, "subcycl": 7, "fourth": 7, "correct": 7, "final": 7, "fifth": 7, "santand": 7, "gijon": 7, "madrid": 7, "sevilla": 7, "barcelona": 7, "bilbao": [7, 15], "valencia": 7, "vigo": 7, "coordin": [7, 14], "43": [7, 17], "4623": 7, "5321": 7, "6644": 7, "40": [7, 11, 12, 15], "4168": 7, "7038": 7, "37": 7, "3891": 7, "9845": 7, "41": [7, 12], "3851": 7, "1734": 7, "2630": 7, "9340": 7, "39": 7, "4699": 7, "3763": 7, "42": 7, "2406": 7, "7207": 7, "204": 7, "426": 7, "916": 7, "724": 7, "111": [7, 13, 16, 17], "605": 7, "435": 7, "429": 7, "980": 7, "944": 7, "174": 7, "690": 7, "348": 7, "536": 7, "620": 7, "394": 7, "353": 7, "824": 7, "995": 7, "1063": 7, "537": 7, "741": 7, "612": 7, "350": 7, "1133": 7, "519": 7, "602": 7, "688": 7, "map": 7, "geograph": 7, "spain": 7, "zoom_start": 7, "coord": 7, "tooltip": 7, "add_to": 7, "popup": 7, "polylin": 7, "km": 7, "plum": [7, 13, 14, 15], "make": [7, 17], "notebook": [7, 12, 13, 15, 16], "trust": [7, 15, 16], "file": [7, 15], "dist": [7, 16], "p": 7, "append": [7, 17], "node": 7, "list": [7, 15], "vertic": 7, "arc": 7, "rout": 7, "have": [7, 17], "sequenti": 7, "plan": 7, "leav": 7, "enter": 7, "58": [7, 14], "63": 7, "238": [7, 14], "45773e": 7, "05": [7, 15, 16], "9616e": 7, "07": [7, 12, 15], "52": 7, "96": [7, 15], "142": [7, 17], "clp0029i": 7, "pass": 7, "2783": 7, "5556": 7, "clp0014i": 7, "perturb": 7, "001": 7, "95238197": 7, "largest": 7, "nonzero": 7, "1344061e": 7, "0011299797": 7, "zero": [7, 12], "7990915e": 7, "555556": 7, "012": 7, "check": 7, "wa": 7, "num_solut": 7, "write": 7, "optimal_rout": 7, "optimal_id": 7, "nc": 7, "break": 7, "3255": 7, "layer": 7, "show": [7, 15, 16], "darkmagenta": 7, "linearconstraint": [8, 10], "nonlinearconstraint": [8, 9, 11], "codes_nl": [8, 9], "plot_packag": [8, 9], "place": 8, "anoth": 8, "contain": 8, "dimens": [8, 9, 13, 14, 15], "trip": 8, "price": 8, "materi": 8, "top": 8, "bottom": 8, "wall": 8, "two": 8, "front": 8, "back": 8, "cubic": 8, "meter": 8, "height": [8, 9], "xy": 8, "2xz": 8, "2yz": 8, "xyz": [8, 9], "3xy": 8, "def": [8, 9, 10, 11, 12], "proport": 8, "constant": 8, "here": 8, "return": [8, 9, 10, 11, 12], "guess": [8, 9, 10, 11, 12], "dim": [8, 9], "set_zlabel": [8, 13, 16, 17], "choos": [8, 9, 11, 17], "slsqp": [8, 9, 11, 12], "1395155369606": 8, "573e": 8, "714e": 8, "572e": 8, "47": 8, "jac": [8, 9, 10, 11], "579e": 8, "03": [8, 15], "995e": 8, "04": [8, 14, 15], "594e": 8, "nfev": [8, 9, 10, 11], "226": 8, "njev": [8, 9, 10, 11], "round": [8, 9, 10], "box": 9, "meet": 9, "requir": 9, "accept": 9, "post": 9, "offic": 9, "perimet": 9, "base": 9, "cannot": 9, "108": 9, "cm": 9, "volum": 9, "nonlinear": 9, "const": 9, "11664": 9, "000078554098": 9, "800e": 9, "600e": 9, "480e": 9, "240e": 9, "73": 9, "36": 9, "l_const": 10, "639": 10, "9999999983455": 10, "anywher": 11, "sketch_vigas_nonlinear": 11, "x_4": [11, 12], "25": [11, 12, 14, 15], "summar": 11, "paramet": [11, 12, 13, 14], "constraint_tf": 11, "nl_const_tf": 11, "1000": [11, 12, 17], "constraint_t": 11, "nl_const_t": 11, "constraint_td": 11, "nl_const_td": 11, "constraint_tb": 11, "nl_const_tb": 11, "15000": 11, "constraint_ta": 11, "nl_const_ta": 11, "x4": 11, "699": 11, "9999999995696": 11, "built": 11, "0x7fce69be7a80": 11, "ejemplo": 12, "optimizaci\u00f3n": 12, "textbf": [12, 13, 16, 17], "textit": 12, "x_1x_2x_3x_4": 12, "x_0": 12, "dond": 12, "se": 12, "busca": 12, "minimizar": 12, "la": 12, "funci\u00f3n": 12, "sujeta": 12, "un": 12, "par": 12, "condicion": 12, "d\u00f3nde": 12, "tienen": 12, "uno": 12, "rango": 12, "determinado": 12, "para": 12, "lo": 12, "par\u00e1metro": 12, "supuesto": 12, "inici": 12, "que": 12, "pued": 12, "utilizar": 12, "hacer": 12, "el": 12, "algoritmo": 12, "busqu": 12, "m\u00e1": 12, "r\u00e1pido": 12, "primero": 12, "utilizar\u00e1": 12, "librer\u00eda": 12, "http": [12, 16, 17], "doc": [12, 17], "org": [12, 15, 16, 17], "refer": [12, 17], "html": [12, 15, 16, 17], "modul": [12, 16], "math": 12, "algorithm": [12, 13, 14], "sce_algorithm": 12, "sce_functioncal": 12, "plotli": 12, "graph_object": 12, "objetivo": 12, "imponen": 12, "restriccion": 12, "llama": 12, "constraint1": 12, "constraint2": 12, "inicial": 12, "valor": 12, "optimizar": 12, "presenta": 12, "sin": [12, 17], "definimo": 12, "b\u00fasqueda": 12, "con1": 12, "type": 12, "ineq": 12, "con2": 12, "eq": 12, "ejecutamo": 12, "soluci\u00f3n": 12, "sol": 12, "\u00f3ptimo": 12, "son": 12, "m\u00ednimo": 12, "74299607": 12, "82115466": 12, "37940764": 12, "01401724556073": 12, "py": [12, 13, 15], "contien": 12, "desarrollado": 12, "por": 12, "q": 12, "duan": 12, "2004": 12, "convert": [12, 16], "python": 12, "van": 12, "hoei": 12, "2011": 12, "stijnvanhoei": 12, "optimization_sc": 12, "muestran": 12, "en": 12, "una": 12, "evalobjf": 12, "quiera": 12, "dependiendo": 12, "del": 12, "testnr": 12, "evaluar": 12, "su": 12, "esta": 12, "vez": 12, "igualdad": 12, "forma": 12, "diferent": 12, "tenemo": 12, "plai": 12, "insertar": 12, "sceua": 12, "script": 12, "sqrt": [12, 17], "modelo": 12, "concultar": 12, "significado": 12, "ise": 12, "iniflg": 12, "ng": 12, "pep": 12, "0001": 12, "maxn": 12, "10000": [12, 14], "kstop": 12, "pcento": 12, "nuevo": 12, "estimaci\u00f3n": 12, "bl": 12, "ones": 12, "bu": 12, "previament": 12, "definido": 12, "buscamo": 12, "bestx": 12, "bestf": 12, "ical": 12, "18453711": 12, "96273555": 12, "55079398": 12, "16618602": 12, "947612187003628": 12, "deben": 12, "cumplir": 12, "debe": 12, "ser": 12, "mayor": 12, "6576160304274339": 12, "105427357601002e": 12, "487420651206776e": 12, "08": [12, 15], "242878379860485e": 12, "figur": [12, 13, 14, 15, 16, 17], "211": 12, "xlabel": [12, 13, 17], "call": 12, "ylabel": [12, 13, 17], "212": 12, "resuelto": 12, "primer": 12, "problema": 12, "ahora": 12, "vamo": 12, "abordar": 12, "mucho": 12, "compleja": 12, "verdad": 12, "evaluar\u00e1n": 12, "eficiencia": 12, "est\u00e1n": 12, "utilizando": 12, "ello": 12, "intentar\u00e1n": 12, "distinta": 12, "global": 12, "optimum": 12, "subplot_kw": 12, "project": [12, 13, 16, 17], "3d": [12, 16, 17], "surf": 12, "plot_surfac": [12, 13, 16, 17], "antialias": 12, "shrink": [12, 16], "92": [12, 13, 14, 16, 17], "bfg": 12, "parametro": 12, "4000036": 12, "99955328": 12, "99910674": 12, "995631096431853e": 12, "99999765": 12, "99999528": 12, "66924042560432e": 12, "origin": 12, "4000": 12, "co": 12, "101": [12, 17], "0000001": 12, "20000": 12, "iteracion": 12, "n_retri": 12, "loss_funct": 12, "best_loss_funct": 12, "999999": 12, "best_paramet": 12, "retri": 12, "5598821274167924": 12, "stack": 12, "std": 12, "0x7fefec85d760": 12, "26019938": 12, "0691066": 12, "44132331792817525": 12, "86108012e": 12, "09": 12, "19006754e": 12, "suptitl": 12, "evolut": 12, "c\u00f3mo": 12, "observar": 12, "consigu": 12, "al": 12, "sencilla": 12, "pero": 12, "respecta": 12, "capaz": 12, "alcanzar": 12, "mientra": 12, "s\u00ed": 12, "obtien": 12, "xarrai": [13, 14, 15], "xr": [13, 15], "io": [13, 15], "loadmat": [13, 15], "gridspec": [13, 14, 15], "same": 13, "folder": 13, "maxdiss_simplified_nothreshold": [13, 14], "normal": [13, 14], "matrix_mda": [13, 14], "vstack": [13, 14, 15, 17], "scalar": [13, 14, 15], "direct": [13, 14, 15], "n_subset": [13, 14, 15], "ix_scalar": [13, 14, 15], "ix_direct": [13, 14, 15], "sel": [13, 14], "maxdiss": [13, 14], "wave": [13, 14, 15], "centroid": 13, "v1": [13, 14, 15], "v1_l": [13, 14, 15], "data": [13, 14], "v2": [13, 14, 15], "v2_l": [13, 14, 15], "tight_layout": [13, 14, 15], "gs": [13, 14, 15], "add_subplot": [13, 14, 15, 16, 17], "170": [13, 15], "grei": [13, 14, 16, 17], "order": 13, "matrix_mda2": 13, "vn": [13, 15], "v3": [13, 14, 15], "v3_l": [13, 14, 15], "60": [13, 17], "ax1": [13, 14, 15, 16, 17], "ax2": [13, 14, 15, 16], "line2d": 13, "0x7fab95fdea60": 13, "n_disc": [13, 16, 17], "500": [13, 14], "discret": [13, 16, 17], "xp": [13, 16, 17], "yp": [13, 16, 17], "reshap": [13, 16, 17], "exp": 13, "zp": [13, 16, 17], "250000": 13, "121": [13, 16, 17], "122": [13, 16, 17], "70": [13, 15, 16], "edgecolor": [13, 15, 16, 17], "stat": 14, "qmc": 14, "name_dim": 14, "hs": 14, "tp": 14, "dir": 14, "n_dim": 14, "lower_bound": 14, "upper_bound": 14, "360": 14, "n_sampl": 14, "combin": [14, 17], "sampler": 14, "latinhypercub": 14, "dataset": [14, 16], "random": [14, 16, 17], "scale": 14, "to_xarrai": 14, "lt": 14, "gt": 14, "int64": 14, "9993": 14, "9994": 14, "9995": 14, "9996": 14, "9997": 14, "9998": 14, "9999": 14, "985": 14, "161": 14, "143": 14, "327": 14, "579": 14, "754": 14, "853": 14, "071": 14, "85": 14, "24": [14, 15], "199": 14, "613": 14, "19": [14, 15], "80": [14, 15, 16, 17], "86": 14, "271": 14, "330": 14, "581": 14, "290": [14, 15], "346": 14, "7xarrai": 14, "datasetdimens": 14, "10000coordin": 14, "int640": 14, "9999arrai": 14, "float649": 14, "071arrai": 14, "98523279": 14, "16105547": 14, "14336118": 14, "75350569": 14, "85288676": 14, "07083798": 14, "float6421": 14, "64arrai": 14, "85497546": 14, "92141363": 14, "19866867": 14, "20121796": 14, "30184627": 14, "63910352": 14, "float6480": 14, "7arrai": 14, "85910799": 14, "89447284": 14, "07592511": 14, "289": 14, "97737115": 14, "23662168": 14, "73750933": 14, "indexpandasindexpandasindex": 14, "rangeindex": 14, "stop": 14, "x27": 14, "attribut": 14, "point": [14, 15, 16, 17], "common": 15, "mpl_toolkit": 15, "mplot3d": 15, "axes3d": 15, "sklearn": [15, 16], "insid": 15, "matrix_kmean": 15, "n_cluster": 15, "random_st": 15, "fit": [15, 16, 17], "In": [15, 16], "jupyt": [15, 16], "environ": [15, 16], "pleas": [15, 16], "rerun": [15, 16], "cell": [15, 16], "represent": [15, 16], "On": [15, 16], "github": [15, 16], "unabl": [15, 16], "render": [15, 16], "page": [1, 2, 3, 4, 5, 15, 16], "nbviewer": [15, 16], "kmeanskmean": 15, "bmu": 15, "labels_": 15, "cluster_centers_": 15, "matrix_kmeans2": 15, "kmeans2": 15, "bmus2": 15, "centroids2": 15, "p_db": 15, "join": 15, "getcwd": 15, "buoi": 15, "databas": 15, "mat": 15, "p_dat": 15, "vizcaya": 15, "ext": 15, "explor": 15, "data_mat": 15, "year": 15, "month": 15, "dai": 15, "hour": 15, "to_datetim": 15, "drop": 15, "set_index": 15, "1990": 15, "98": 15, "112": 15, "104": 15, "2009": 15, "309": 15, "310": 15, "59119": 15, "data_norm": 15, "mini": 15, "maxi": 15, "07352941": 15, "11764706": 15, "53333333": 15, "14973262": 15, "54444444": 15, "06617647": 15, "62222222": 15, "125": [15, 17], "27272727": 15, "71666667": 15, "72222222": 15, "kma": 15, "n_init": 15, "centroids_norm": 15, "values_norm": 15, "_": 15, "bmus_pr": 15, "int32": 15, "58158568": 15, "02966752": 15, "303": 15, "00255754": 15, "23842365": 15, "89507389": 15, "196": 15, "neighbor": 16, "kneighborsregressor": 16, "preprocess": 16, "scikit": 16, "learn": 16, "stabl": 16, "gener": [16, 17], "abov": [1, 2, 3, 4, 5, 16, 17], "df": 16, "n_random": [16, 17], "chooss": [16, 17], "df_sel": 16, "sampl": 16, "reset_index": 16, "n_analogu": 16, "min_max_scal": 16, "minmaxscal": 16, "x_train_minmax": 16, "fit_transform": 16, "neigh": 16, "n_neighbor": 16, "kneighborsregressorkneighborsregressor": 16, "i_an": 16, "kneighbor": 16, "return_dist": 16, "multipli": 16, "factor": 16, "invers": 16, "transpos": 16, "5850": 16, "royalblu": 16, "r": 16, "loc": 16, "0x7f7bb764d940": 16, "To": 16, "its": 16, "z_rec": 16, "z_rec_p": 16, "131": 16, "orient": 16, "pad": 16, "133": 16, "lim": 16, "nanmax": 16, "ab": 16, "nanmin": 16, "rdbu_r": [16, 17], "delta": 16, "interpol": 17, "rbfinterpol": 17, "radial": 17, "basi": 17, "test": 17, "random_i": 17, "randint": 17, "xi": 17, "yi": 17, "zi": 17, "surfac": 17, "2d": 17, "im1": 17, "rbf_func": 17, "instanc": 17, "With": 17, "also": 17, "error": 17, "ax0": 17, "now": 17, "larger": 17, "rmat": 17, "250": 17, "astyp": 17, "int": 17, "26": 17, "34": 17, "51": 17, "59": 17, "67": 17, "76": 17, "84": 17, "109": 17, "117": 17, "134": 17, "158": 17, "167": 17, "175": 17, "183": 17, "192": 17, "208": 17, "216": 17, "225": 17, "233": 17, "241": 17, "rmse": 17, "section": [], "describ": [], "bring": [], "interact": [], "your": [], "book": [], "user": [], "run": [1, 2, 3, 4, 5], "code": [1, 2, 3, 4, 5], "output": [], "without": [], "provid": [], "kernel": [], "public": [], "mybind": [], "servic": [], "click": [1, 2, 3, 4, 5], "live": [1, 2, 3, 4, 5], "button": [1, 2, 3, 4, 5], "fa": 4, "rocket": 4, "guilabel": 4}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"m2248": 0, "optim": [0, 3, 4, 12, 17], "civil": 0, "engin": 0, "summari": 0, "linear": [1, 10], "program": 1, "transport": [2, 8], "problem": [2, 6, 12], "simplex": [3, 4], "method": [3, 4], "ex1": 3, "1": [3, 4], "standar": [3, 4], "form": [3, 4], "2": [3, 4], "determin": [3, 4], "slack": [3, 4], "variabl": [3, 4], "3": [3, 4], "set": [3, 4], "up": [3, 4], "tableau": [3, 4], "4": [3, 4], "check": [3, 4], "5": [3, 4], "identifi": [3, 4], "pivot": [3, 4], "6": [3, 4], "creat": [3, 4, 15], "new": [3, 4], "repeat": [3, 4], "ex2": 4, "string": [5, 10, 11], "beam": [5, 10, 11], "system": [5, 10, 11], "The": [6, 7], "knapsack": 6, "travel": 7, "salesman": 7, "sand": 8, "postal": 9, "packag": 9, "nonlinear": 11, "heurist": 12, "implementaci\u00f3n": 12, "con": 12, "scipi": 12, "sce": 12, "ua": 12, "validamo": 12, "resultado": 12, "more": 12, "rosenbrock": 12, "function": 12, "griewank": 12, "resumen": 12, "de": 12, "y": 12, "grienwank": 12, "select": [13, 14], "mda": [13, 14], "2d": [13, 15], "3d": [13, 15], "point": 13, "over": 13, "surfac": [13, 16], "sampl": 14, "lh": 14, "plot": 14, "centroid": [14, 15], "cluster": [15, 17], "kmean": 15, "real": 15, "data": 15, "matrix": 15, "normal": 15, "calcul": 15, "denorm": 15, "predict": 15, "given": 15, "valu": 15, "hs": 15, "tp": 15, "dir": 15, "reconstruct": [16, 17], "analogu": 16, "rbf": 17, "mexican": 17, "hat": 17, "number": 17, "thebe": [], "test": []}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx": 56}})