import torch
import torch.nn.functional as F
from timeit import default_timer


def train_fno(model, myloss, epochs, batch_size, train_loader, test_loader,
              optimizer, scheduler, normalized, normalizer, device):
    train_mse_log = []
    train_l2_log = []
    test_l2_log = []

    if normalized:
        a_normalizer = normalizer[0].to(device)
        y_normalizer = normalizer[1].to(device)
    else:
        a_normalizer = None
        y_normalizer = None

    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            if normalized:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)

            mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            loss.backward()

            optimizer.step()
            scheduler.step()
            train_mse += mse.item()
            train_l2 += loss.item()

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)

                out = model(x)
                if normalized:
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)

                test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

        train_mse /= len(train_loader)
        train_l2 /= (batch_size * len(train_loader))
        test_l2 /= (batch_size * len(test_loader))

        train_mse_log.append(train_mse)
        train_l2_log.append(train_l2)
        test_l2_log.append(test_l2)

        t2 = default_timer()
        print(ep, t2 - t1, train_mse, train_l2, test_l2)

    return model, train_mse_log, train_l2_log, test_l2_log
