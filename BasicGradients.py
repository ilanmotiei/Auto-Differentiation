
# This dictionary will contain the methods calculate jacobian-vector-products efficiently,
# right after initialized by "__init__".

jvp_dict = {}


def defjvp(key, *functions):

    if len(functions) == 1:
        jvp_dict[key] = functions[0]
    else:
        jvp_dict[key] = functions
