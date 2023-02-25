import re
import sys
import yaml
from enum import Enum


class Config(object):

    def __init__(self, config_file_list=None, config_dict=None):
        """
        Args:
            config_file_list (list of str): the external config file, it allows multiple config files, default is None.
            config_dict (dict): the external parameter dictionaries, default is None.
        """
        self._init_parameters_category()
        self.yaml_loader = self._build_yaml_loader()
        self.file_config_dict = self._load_config_files(config_file_list)
        self.variable_config_dict = self._load_variable_config_dict(config_dict)
        self.cmd_config_dict = self._load_cmd_line()
        self._merge_external_config_dict()
        self.final_config_dict = self._get_final_config_dict()

    def _init_parameters_category(self):
        self.parameters = dict()

    def _build_yaml_loader(self):
        loader = yaml.FullLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(
                u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X
            ), list(u'-+0123456789.')
        )
        return loader

    def _convert_config_dict(self, config_dict):
        r"""This function convert the str parameters to their original type.

        """
        for key in config_dict:
            param = config_dict[key]
            if not isinstance(param, str):
                continue
            try:
                value = eval(param)
                if value is not None and not isinstance(value, (str, int, float, list, tuple, dict, bool, Enum)):
                    value = param
            except (NameError, SyntaxError, TypeError):
                if isinstance(param, str):
                    if param.lower() == "true":
                        value = True
                    elif param.lower() == "false":
                        value = False
                    else:
                        value = param
                else:
                    value = param
            config_dict[key] = value
        return config_dict

    def _load_config_files(self, file_list):
        file_config_dict = dict()
        if file_list:
            for file in file_list:
                with open(file, 'r', encoding='utf-8') as f:
                    file_config_dict.update(yaml.load(f.read(), Loader=self.yaml_loader))
        return file_config_dict

    def _load_variable_config_dict(self, config_dict):
        # HyperTuning may set the parameters such as mlp_hidden_list in the format of ['[]', '[]']
        # then config_dict will receive a str '[]', but indeed it's a list []
        # temporarily use _convert_config_dict to solve this problem
        return self._convert_config_dict(config_dict) if config_dict else dict()

    def _load_cmd_line(self):
        r""" Read parameters from command line and convert it to str.

        """
        cmd_config_dict = dict()
        unrecognized_args = []
        if "ipykernel_launcher" not in sys.argv[0]:
            for arg in sys.argv[1:]:
                if not arg.startswith("--") or len(arg[2:].split("=")) != 2:
                    unrecognized_args.append(arg)
                    continue
                cmd_arg_name, cmd_arg_value = arg[2:].split("=")
                if cmd_arg_name in cmd_config_dict and cmd_arg_value != cmd_config_dict[cmd_arg_name]:
                    raise SyntaxError("There are duplicate commend arg '%s' with different value." % arg)
                else:
                    cmd_config_dict[cmd_arg_name] = cmd_arg_value
        if len(unrecognized_args) > 0:
            print('command line args [{}] will not be used'.format(' '.join(unrecognized_args)))

        cmd_config_dict = self._convert_config_dict(cmd_config_dict)
        return cmd_config_dict

    def _merge_external_config_dict(self):
        external_config_dict = dict()
        external_config_dict.update(self.file_config_dict)
        external_config_dict.update(self.variable_config_dict)
        external_config_dict.update(self.cmd_config_dict)
        self.external_config_dict = external_config_dict

    def _get_final_config_dict(self):
        final_config_dict = dict()
        final_config_dict.update(self.external_config_dict)
        return final_config_dict
