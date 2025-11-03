import torch
import torch.nn as nn
from .transformer_flux import FluxTransformer2DModel

class FluxNetwork(nn.Module):
    TARGET_REPLACE_MODULE = ["FluxTransformerBlock","FluxSingleTransformerBlock"] # 可训练的模块类型
    FLUX_PREFIX = "flux"
    
    def __init__(self, flux_model: FluxTransformer2DModel):
        super().__init__()
        self.flux_model = flux_model
        self.trainable_component_names = []  # 用于记录可训练组件的名称
      
    @staticmethod
    def generate_trainable_components(layers, num_transformer_blocks=19):
        transformer_components = [
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out",
            "norm1",
            "norm1_context",
        ]
        
        single_transformer_components = [
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "norm",
            #"proj_mlp",
        ]
        
        components = ["context_embedder"]  # 添加 context_embedder
        for layer in layers:
            if layer < num_transformer_blocks:
                prefix = f"transformer_blocks.{layer}"
                base_components = transformer_components
            else:
                prefix = f"single_transformer_blocks.{layer - num_transformer_blocks}"
                base_components = single_transformer_components
            components.extend([f"{prefix}.{comp}" for comp in base_components])
        
        return components
    
    #def apply_to(self, num_layers=1, additional_components=None):
    #    component_names = self.generate_trainable_components(num_layers)
    #    
    #    if additional_components:
    #        component_names.extend(additional_components)
    #    
    #    self.trainable_component_names = []  # 重置
    #    for name in component_names:
    #        recursive_getattr(self.flux_model, name).requires_grad_(True)
    #        self.trainable_component_names.append(name)  # 记录名称
    
    #def apply_to(self, num_layers=1, additional_components=None):
    #    component_names = self.generate_trainable_components(num_layers)
    #    
    #    if additional_components:
    #        component_names.extend(additional_components)
    #    
    #    self.trainable_component_names = []  # 重置
    #    for name in component_names:
    #        component = recursive_getattr(self.flux_model, name)
    #        if isinstance(component, nn.Module):
    #            component.requires_grad_(True)
    #            self.trainable_component_names.append(name)
    #        else:
    #            print(f"Warning: {name} is not a Module, skipping.")
    
    def apply_to(self, layers=None, additional_components=None):
        if layers is None:
            layers = list(range(57))  # 默认包含所有层
        
        component_names = self.generate_trainable_components(layers)
        
        if additional_components:
            component_names.extend(additional_components)
        
        self.trainable_component_names = []  # 重置
        for name in component_names:
            try:
                component = recursive_getattr(self.flux_model, name)
                if isinstance(component, nn.Module):
                    component.requires_grad_(True)
                    self.trainable_component_names.append(name)
                else:
                    print(f"Warning: {name} is not a Module, skipping.")
            except AttributeError:
                print(f"Warning: {name} not found in the model, skipping.")
                        
    def prepare_grad_etc(self):
        # 供flux_model调用,用于冻结/解冻组件
        self.flux_model.requires_grad_(False)
        for name in self.trainable_component_names:
            recursive_getattr(self.flux_model, name).requires_grad_(True)
                
    def get_trainable_params(self):
        # 返回需要训练的参数
        params = []
        for name in self.trainable_component_names:
            params.extend(recursive_getattr(self.flux_model, name).parameters())
        return params
    
    def print_trainable_params_info(self):
        total_params = 0
        for name in self.trainable_component_names:
            module = recursive_getattr(self.flux_model, name)
            module_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += module_params
            #print(f'{name}: {module_params} trainable parameters')
        print(f'Total trainable params: {total_params}')
    
    def save_weights(self, file, dtype=None):
        # 保存需要训练的组件参数
        state_dict = {}
        for name in self.trainable_component_names:
            state_dict[name] = recursive_getattr(self.flux_model, name).state_dict()
        if dtype is not None:
            for v in state_dict.values():
                v = {k: t.detach().clone().to("cpu").to(dtype) for k, t in v.items()}        
        torch.save(state_dict, file)

    #def load_weights(self, file):    
    #    # 加载需要训练的组件参数
    #    state_dict = torch.load(file, weights_only=True)
    #    for name in state_dict:
    #        module = recursive_getattr(self.flux_model, name)
    #        module.load_state_dict(state_dict[name])
    #        print(f"加载参数: {name}")
    
    def load_weights(self, file, device):
        print(f"Loading weights from {file}")
        try:
            state_dict = torch.load(file, map_location=device, weights_only=True)
        except Exception as e:
            print(f"Failed to load weights from {file}: {str(e)}")
            return False

        successfully_loaded = []
        failed_to_load = []

        for name in state_dict:
            try:
                module = recursive_getattr(self.flux_model, name)
                module_state_dict = module.state_dict()
                
                # 检查state_dict的键是否匹配
                if set(state_dict[name].keys()) != set(module_state_dict.keys()):
                    raise ValueError(f"State dict keys for {name} do not match")
                
                # 检查张量的形状是否匹配
                for key in state_dict[name]:
                    if state_dict[name][key].shape != module_state_dict[key].shape:
                        raise ValueError(f"Shape mismatch for {name}.{key}")
                
                module.load_state_dict(state_dict[name])
                successfully_loaded.append(name)
                 
            except Exception as e:
                print(f"Failed to load weights for {name}: {str(e)}")
                failed_to_load.append(name)

        if successfully_loaded:
            print(f"Successfully loaded weights for: {', '.join(successfully_loaded)}")
        if failed_to_load:
            print(f"Failed to load weights for: {', '.join(failed_to_load)}")

        return len(failed_to_load) == 0  # 如果没有加载失败的组件，则返回True
            
# 改进的递归获取属性函数
def recursive_getattr(obj, attr):
    attrs = attr.split(".")
    for i in range(len(attrs)):
        obj = getattr(obj, attrs[i]) 
    return obj

# 递归设置属性函数
def recursive_setattr(obj, attr, val):
    attrs = attr.split(".")
    for i in range(len(attrs)-1):
        obj = getattr(obj, attrs[i])
    setattr(obj, attrs[-1], val)