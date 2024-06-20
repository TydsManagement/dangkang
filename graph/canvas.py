#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import importlib
import json
import traceback
from abc import ABC
from copy import deepcopy
from functools import partial

import pandas as pd

from graph.component import component_class
from graph.component.base import ComponentBase
from graph.settings import flow_logger, DEBUG


class Canvas(ABC):
    """
    dsl = {
        "components": {
            "begin": {
                "obj":{
                    "component_name": "Begin",
                    "params": {},
                },
                "downstream": ["answer_0"],
                "upstream": [],
            },
            "answer_0": {
                "obj": {
                    "component_name": "Answer",
                    "params": {}
                },
                "downstream": ["retrieval_0"],
                "upstream": ["begin", "generate_0"],
            },
            "retrieval_0": {
                "obj": {
                    "component_name": "Retrieval",
                    "params": {}
                },
                "downstream": ["generate_0"],
                "upstream": ["answer_0"],
            },
            "generate_0": {
                "obj": {
                    "component_name": "Generate",
                    "params": {}
                },
                "downstream": ["answer_0"],
                "upstream": ["retrieval_0"],
            }
        },
        "history": [],
        "messages": [],
        "reference": [],
        "path": [["begin"]],
        "answer": []
    }
    """

    def __init__(self, dsl: str, tenant_id=None):
        self.path = []
        self.history = []
        self.messages = []
        self.answer = []
        self.components = {}
        self.dsl = json.loads(dsl) if dsl else {
            "components": {
                "begin": {
                    "obj": {
                        "component_name": "Begin",
                        "params": {
                            "prologue": "Hi there!"
                        }
                    },
                    "downstream": [],
                    "upstream": []
                }
            },
            "history": [],
            "messages": [],
            "reference": [],
            "path": [],
            "answer": []
        }
        self._tenant_id = tenant_id
        self._embed_id = ""
        self.load()

    def load(self):
        assert self.dsl.get("components", {}).get("begin"), "There have to be a 'Begin' component."

        self.components = self.dsl["components"]
        for k, cpn in self.components.items():
            param = component_class(cpn["obj"]["component_name"] + "Param")()
            param.update(cpn["obj"]["params"])
            param.check()
            cpn["obj"] = component_class(cpn["obj"]["component_name"])(self, k, param)
            if cpn["obj"].component_name == "Categorize":
                for _,desc in param.category_description.items():
                    if desc["to"] not in cpn["downstream"]:
                        cpn["downstream"].append(desc["to"])

        self.path = self.dsl["path"]
        self.history = self.dsl["history"]
        self.messages = self.dsl["messages"]
        self.answer = self.dsl["answer"]
        self.reference = self.dsl["reference"]
        self._embed_id = self.dsl.get("embed_id", "")

    def __str__(self):
        self.dsl["path"] = self.path
        self.dsl["history"] = self.history
        self.dsl["messages"] = self.messages
        self.dsl["answer"] = self.answer
        self.dsl["reference"] = self.reference
        self.dsl["embed_id"] = self._embed_id
        dsl = deepcopy(self.dsl)
        for k, cpn in self.components.items():
            dsl["components"][k]["obj"] = json.loads(str(cpn["obj"]))
        return json.dumps(dsl, ensure_ascii=False)

    def reset(self):
        self.path = []
        self.history = []
        self.messages = []
        self.answer = []
        self.reference = []
        self.components = {}
        self._embed_id = ""

    def run(self, **kwargs):
        ans = ""
        if self.answer:
            cpn_id = self.answer[0]
            self.answer.pop(0)
            try:
                ans = self.components[cpn_id]["obj"].run(self.history, **kwargs)
            except Exception as e:
                ans = ComponentBase.be_output(str(e))
            self.path[-1].append(cpn_id)
            self.history.append(("assistant", ans.to_dict("records")))
            return ans

        if not self.path:
            self.components["begin"]["obj"].run(self.history, **kwargs)
            self.path.append(["begin"])

        self.path.append([])
        ran = -1

        def prepare2run(cpns):
            nonlocal ran, ans
            for c in cpns:
                cpn = self.components[c]["obj"]
                if cpn.component_name == "Answer":
                    self.answer.append(c)
                else:
                    if DEBUG: print("RUN: ", c)
                    ans = cpn.run(self.history, **kwargs)
                    self.path[-1].append(c)
                ran += 1

        prepare2run(self.components[self.path[-2][-1]]["downstream"])
        while ran < len(self.path[-1]):
            if DEBUG: print(ran, self.path)
            cpn_id = self.path[-1][ran]
            cpn = self.get_component(cpn_id)
            if not cpn["downstream"]: break

            if cpn["obj"].component_name.lower() in ["switch", "categorize", "relevant"]:
                switch_out = cpn["obj"].output()[1].iloc[0, 0]
                assert switch_out in self.components, \
                    "{}'s output: {} not valid.".format(cpn_id, switch_out)
                try:
                    prepare2run([switch_out])
                except Exception as e:
                    for p in [c for p in self.path for c in p][::-1]:
                        if p.lower().find("answer") >= 0:
                            self.get_component(p)["obj"].set_exception(e)
                            prepare2run([p])
                            break
                    traceback.print_exc()
                continue

            try:
                prepare2run(cpn["downstream"])
            except Exception as e:
                for p in [c for p in self.path for c in p][::-1]:
                    if p.lower().find("answer") >= 0:
                        self.get_component(p)["obj"].set_exception(e)
                        prepare2run([p])
                        break
                traceback.print_exc()

        if self.answer:
            cpn_id = self.answer[0]
            self.answer.pop(0)
            ans = self.components[cpn_id]["obj"].run(self.history, **kwargs)
            self.path[-1].append(cpn_id)
            if kwargs.get("stream"):
                assert isinstance(ans, partial)
                return ans

            self.history.append(("assistant", ans.to_dict("records")))

        return ans

    def get_component(self, cpn_id):
        return self.components[cpn_id]

    def get_tenant_id(self):
        return self._tenant_id

    def get_history(self, window_size):
        convs = []
        for role, obj in self.history[window_size * -2:]:
            convs.append({"role": role, "content": (obj if role == "user" else
                                                    '\n'.join(pd.DataFrame(obj)['content']))})
        return convs

    def add_user_input(self, question):
        self.history.append(("user", question))

    def set_embedding_model(self, embed_id):
        self._embed_id = embed_id

    def get_embedding_model(self):
        return self._embed_id
