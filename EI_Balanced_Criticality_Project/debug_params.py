import brainpy as bp
import brainpy.math as bm
# 注意：这里完全模仿你项目的导入路径
from configs.model_config import (
    G_EE, G_IE, G_EXT_E, EXT_FREQ_TOTAL, 
    N_E, CONN_PROB
)
from models.network import BalancedNetwork

def detect_parameters():
    print("\n" + "="*60)
    print("🕵️ PARAMETER DETECTIVE: What is the code actually seeing?")
    print("="*60)

    # 1. 打印静态配置参数
    print(f"\n[1] Check Config Variables (Static):")
    print(f"    G_EE (Recurrent E->E) : {G_EE:.6f}")
    print(f"    G_IE (Recurrent I->E) : {G_IE:.6f}")
    print(f"    G_EXT_E (External->E) : {G_EXT_E:.6f}")
    print(f"    EXT_FREQ_TOTAL        : {EXT_FREQ_TOTAL}")
    
    # 理论预期值 (W_SCALE = 0.6 时)
    # G_EE 应该是 0.012 * 0.6 = 0.0072
    # G_IE 应该是 0.18  * 0.6 = 0.1080
    
    if abs(G_EE - 0.0072) < 0.0001:
        print("    ✅ W_SCALE=0.6 seems ACTIVE.")
    elif abs(G_EE - 0.012) < 0.0001:
        print("    ❌ W_SCALE is NOT ACTIVE! (Reading original 0.012)")
        print("    👉 You are likely editing the wrong config file!")
    else:
        print(f"    ⚠️ Unknown scaling. (G_EE={G_EE})")

    # 2. 打印网络实例中的实际参数
    print(f"\n[2] Check Network Instance (Runtime):")
    net = BalancedNetwork(tau_d_I=8.0)
    
    # 获取突触对象 (BrainPy 的 Synapse 对象通常存储了 g_max)
    # 注意：这取决于 DualExpCondSyn 的实现，通常它会有一个 g_max 属性
    try:
        real_g_ee = net.E2E.g_max
        real_g_ie = net.I2E.g_max
        
        # 如果 g_max 是 brainpy Array，取第一个值
        if isinstance(real_g_ee, (bm.Array, bp.Array)):
            real_g_ee = real_g_ee[0] if real_g_ee.ndim > 0 else real_g_ee
        if isinstance(real_g_ie, (bm.Array, bp.Array)):
            real_g_ie = real_g_ie[0] if real_g_ie.ndim > 0 else real_g_ie
            
        print(f"    net.E2E.g_max         : {float(real_g_ee):.6f}")
        print(f"    net.I2E.g_max         : {float(real_g_ie):.6f}")
        
    except AttributeError:
        print("    ⚠️ Could not access g_max directly on synapse object.")

    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    detect_parameters()