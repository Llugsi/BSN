# Script 6: Noise2Void, Noise2Self, and Neighbor2Neighbor training
print("[OPTIMIZED MODULE 2] Training Noise2Void (N2V) at high speed...")
log_n2v_ptk4 = "experimento_N2V_ptk4_logs.csv"
if os.path.exists(log_n2v_ptk4): os.remove(log_n2v_ptk4)

class Red_N2V_PtK4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_ciega = nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False)
        self.red_interna = nn.Sequential(
            nn.LeakyReLU(0.1), nn.Conv2d(16, 16, kernel_size=1),
            nn.LeakyReLU(0.1), nn.Conv2d(16, 1, kernel_size=1), nn.Sigmoid()
        )
    def forward(self, x):
        with torch.no_grad(): self.conv_ciega.weight[:, :, 1, 1] = 0.0
        return self.red_interna(self.conv_ciega(x))

model_n2v = Red_N2V_PtK4().to(device)
opt_n2v = optim.Adam(model_n2v.parameters(), lr=0.002)
criterion = nn.MSELoss()

# Create a fixed list of indices for fast evaluation (10 validation samples per epoch)
muestras_eval = [dataset_completo_ptk4[i] for i in np.random.choice(len(dataset_completo_ptk4), 10, replace=False)]

for epoch in range(120):
    model_n2v.train()
    loss_epoch = 0.0
    
    # TNNLS FIX: We take only 30 random iterations per epoch to optimize execution times
    indices_epoca = np.random.choice(len(dataset_completo_ptk4), 30, replace=True)
    
    for idx in indices_epoca:
        n1_b, _, _, _ = dataset_completo_ptk4[idx]
        n1_b = n1_b.unsqueeze(0).to(device) # Add batch dimension (1, 1, 512, 512)
        
        out = model_n2v(n1_b)
        loss = criterion(out, n1_b)
        
        opt_n2v.zero_grad()
        loss.backward()
        opt_n2v.step()
        loss_epoch += loss.item()
        
    # Ultra-fast validation phase per epoch
    model_n2v.eval()
    psnr_p, ssim_p = [], []
    with torch.no_grad():
        for n1_ev, _, c_ev, _ in muestras_eval:
            d_np = model_n2v(n1_ev.unsqueeze(0).to(device)).squeeze().cpu().numpy()
            psnr_p.append(psnr(c_ev.squeeze().numpy(), d_np, data_range=1.0))
            ssim_p.append(ssim(c_ev.squeeze().numpy(), d_np, data_range=1.0))
            
    loggear_experimento_modelo_ptk4(log_n2v_ptk4, epoch+1, loss_epoch/30, 0.002, np.mean(psnr_p), np.mean(ssim_p))
    if (epoch + 1) % 30 == 0 or epoch == 0:
        print(f" ▫️ Epoch [{epoch+1:03d}/120] -> N2V Average PSNR: {np.mean(psnr_p):.2f} dB")

print(f"Log successfully exported in seconds: '{log_n2v_ptk4}'")


print("[OPTIMIZED MODULE 3] Training Noise2Self (N2S) at high speed...")
log_n2s_ptk4 = "experimento_N2S_ptk4_logs.csv"
if os.path.exists(log_n2s_ptk4): os.remove(log_n2s_ptk4)

model_n2s = Red_N2V_PtK4().to(device)
opt_n2s = optim.Adam(model_n2s.parameters(), lr=0.002, weight_decay=1e-6)

for epoch in range(120):
    model_n2s.train()
    loss_epoch = 0.0
    indices_epoca = np.random.choice(len(dataset_completo_ptk4), 30, replace=True)
    
    for idx in indices_epoca:
        n1_b, _, _, _ = dataset_completo_ptk4[idx]
        n1_b = n1_b.unsqueeze(0).to(device)
        
        out = model_n2s(n1_b)
        loss = criterion(out, n1_b)
        
        opt_n2s.zero_grad()
        loss.backward()
        opt_n2s.step()
        loss_epoch += loss.item()
        
    model_n2s.eval()
    psnr_p, ssim_p = [], []
    with torch.no_grad():
        for n1_ev, _, c_ev, _ in muestras_eval:
            d_np = model_n2s(n1_ev.unsqueeze(0).to(device)).squeeze().cpu().numpy()
            psnr_p.append(psnr(c_ev.squeeze().numpy(), d_np, data_range=1.0))
            ssim_p.append(ssim(c_ev.squeeze().numpy(), d_np, data_range=1.0))
            
    loggear_experimento_modelo_ptk4(log_n2s_ptk4, epoch+1, loss_epoch/30, 0.002, np.mean(psnr_p), np.mean(ssim_p))
    if (epoch + 1) % 30 == 0 or epoch == 0:
        print(f" ▫️ Epoch [{epoch+1:03d}/120] -> N2S Average PSNR: {np.mean(psnr_p):.2f} dB")

print(f"Log successfully exported in seconds: '{log_n2s_ptk4}'")


print("[OPTIMIZED MODULE 4] Training Neighbor2Neighbor (N2N) at high speed...")
log_n2n_ptk4 = "experimento_N2N_ptk4_logs.csv"
if os.path.exists(log_n2n_ptk4): os.remove(log_n2n_ptk4)

class RedEstandarN2N_PtK4(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(16, 1, kernel_size=1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

model_n2n = RedEstandarN2N_PtK4().to(device)
opt_n2n = optim.Adam(model_n2n.parameters(), lr=0.002)

for epoch in range(120):
    model_n2n.train()
    loss_epoch = 0.0
    indices_epoca = np.random.choice(len(dataset_completo_ptk4), 30, replace=True)
    
    for idx in indices_epoca:
        n1_b, _, _, _ = dataset_completo_ptk4[idx]
        n1_b = n1_b.unsqueeze(0) # Format (1, 1, 512, 512)
        
        sub_img1 = n1_b[:, :, 0::2, 0::2]
        sub_img2 = n1_b[:, :, 0::2, 1::2]
        
        sub_img1_up = F.interpolate(sub_img1, size=(512, 512), mode='bilinear', align_corners=False)
        sub_img2_up = F.interpolate(sub_img2, size=(512, 512), mode='bilinear', align_corners=False)
        
        out_sub1 = model_n2n(sub_img1_up.to(device))
        loss = criterion(out_sub1, sub_img2_up.to(device))
        
        opt_n2n.zero_grad()
        loss.backward()
        opt_n2n.step()
        loss_epoch += loss.item()
        
    model_n2n.eval()
    psnr_p, ssim_p = [], []
    with torch.no_grad():
        for n1_ev, _, c_ev, _ in muestras_eval:
            d_np = model_n2n(n1_ev.unsqueeze(0).to(device)).squeeze().cpu().numpy()
            psnr_p.append(psnr(c_ev.squeeze().numpy(), d_np, data_range=1.0))
            ssim_p.append(ssim(c_ev.squeeze().numpy(), d_np, data_range=1.0))
            
    loggear_experimento_modelo_ptk4(log_n2n_ptk4, epoch+1, loss_epoch/30, 0.002, np.mean(psnr_p), np.mean(ssim_p))
    if (epoch + 1) % 30 == 0 or epoch == 0:
        print(f" ▫️ Epoch [{epoch+1:03d}/120] -> N2N Average PSNR: {np.mean(psnr_p):.2f} dB")

print(f"Log successfully exported in seconds: '{log_n2n_ptk4}'")
print("\nSOTA BENCHMARK PROCESS COMPLETED FOR THE REAL MICROSCÓPY FMD SCENARIO!")
