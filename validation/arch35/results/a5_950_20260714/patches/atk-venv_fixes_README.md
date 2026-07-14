# ATK venv 临时兼容补丁（不回传算子仓）

> ATK 版本: 26.5.14 (atk-26.5.14-py3-none-any.whl)
> venv: /workspace/venv
> 文件路径:
>   /workspace/venv/lib/python3.12/site-packages/atk/tasks/backends/lib_interface/acl_wrapper.py
>   /workspace/venv/lib/python3.12/site-packages/sitecustomize.py
>   /workspace/venv/lib/python3.12/site-packages/atk/tasks/backends/lib_interface/lib_manager.py  (见 lib_manager_PATCHED.py)

这些是 **venv 内 ATK 工具的兼容补丁**，只影响本容器的 ATK 运行，**不提交到
任何算子仓**。最终报告会明确列出。

## acl_wrapper_PATCHED.py（相对 pip 原始 wheel）
两处路径兜底（CANN 9.1.0-beta.1 布局与 ATK 假设的旧 acllib/lib64 不同）:
1. get_opp_lib_path: ASCEND_OPP_PATH/lib64/libopapi.so 找不到时，回退到
   ${ASCEND_TOOLKIT_HOME}/x86_64-linux/lib64/libopapi.so。
   否则整个 try 块抛 FileNotFoundError，ascendcl/aclnn/nnopbase 全部 = None。
2. ASCENDCL_PATH / NNOPBASE_PATH: acllib/lib64 找不到时回退到
   x86_64-linux/lib64。

## sitecustomize.py
numpy 2.x + torch 2.9 的安全限制：torch.load 反序列化 numpy 标量/dtype 时
拒绝 numpy._core.multiarray.scalar / numpy.dtypes.* 等全局。ops-nn 要求
numpy<2.0，但本环境系统 numpy 是 2.4.6 且 PyPI 慢装不上 1.x。sitecustomize
在 Python 启动时把这些 numpy 全局加入 torch safe_globals，对 celery worker
同样生效。

## lib_manager.py（bind_function 回退，详见 lib_manager_PATCHED.py）
ATK pyaclnn 后端在 import 时预加载 aclnn 管理器为 CANN 自带 libopapi.so，但
aclnnGammaAddRmsNorm[GetWorkspaceSize] 符号在自建 libcust_opapi.so 里。
executor 的 get_opp_lib_path 单函数重定向不生效（因为是预加载 manager +
bind_function）。
修复: bind_function 对**仅这两个**符号查主库失败后，回退到 GAMMA_OPAPI_LIB
指向的自定义库（单 handle、CDLL RTLD_GLOBAL、模块级缓存，不重复 dlopen）。
