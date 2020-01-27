import os
import json
import pickle
import torch
from src.modeling.meta_arch.build import build_model
from src.data.bengali_data import build_data_loader
from src.modeling.solver.optimizer import build_optimizer
from src.modeling.solver.evaluation import build_evaluator


def train(cfg):
    # FILES, PATHS
    assert cfg.OUTPUT_PATH != ''
    output_path = cfg.OUTPUT_PATH
    train_path = cfg.DATASET.TRAIN_PATH
    val_path = cfg.DATASET.VAL_PATH

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    backup_dir = os.path.join(output_path, 'model_backups')
    if not os.path.exists(backup_dir):
        os.mkdir(backup_dir)
    state_fpath = os.path.join(output_path, 'model.pt')
    perf_path = os.path.join(output_path, 'trace.json')
    perf_trace = []

    # DATA LOADER
    train_data = pickle.load(open(train_path, 'rb'))
    val_data = pickle.load(open(val_path, 'rb'))
    train_loader = build_data_loader(train_data, cfg.DATASET, True)
    val_loader = build_data_loader(val_data, cfg.DATASET, False)

    # MODEL
    model = build_model(cfg.MODEL)
    current_epoch = 0
    if cfg.RESUME_PATH != "":
        checkpoint = torch.load(cfg.RESUME_PATH, map_location='cpu')
        current_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint["model_state"])
    _ = model.cuda()

    # SOLVER EVALUATOR
    solver_cfg = cfg.SOLVER
    optimizer = build_optimizer(model, solver_cfg)
    evaluator = build_evaluator(solver_cfg)
    total_epochs = solver_cfg.TOTAL_EPOCHS

    for epoch in range(current_epoch, total_epochs):
        model.train()
        print('Start epoch', epoch)
        train_itr = iter(train_loader)
        total_err = 0
        total_acc = 0
        for idx, (inputs, labels) in enumerate(train_itr):

            # compute
            input_data = inputs.cuda()
            labels = labels.cuda()
            grapheme_logits, vowel_logits, consonant_logits = model(input_data)

            eval_result = evaluator(grapheme_logits, vowel_logits, consonant_logits, labels)
            optimizer.zero_grad()
            eval_result['loss'].backward()
            optimizer.step()

            eval_result = {k: eval_result[k].item() for k in eval_result}
            total_err += eval_result['loss']
            total_acc += eval_result['acc']
            if idx % 100 == 0:
                print(idx, eval_result['loss'], eval_result['acc'])

        train_total_err = total_err / (1 + idx)
        train_total_acc = total_acc / (1 + idx)
        print("Epoch {0} Training, Loss {1}, Acc {2}".format(epoch, train_total_err, train_total_acc))

        # compute validation error
        model.eval()
        val_itr = iter(val_loader)
        total_err = 0
        total_acc = 0
        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(val_itr):
                input_data = inputs.cuda()
                labels = labels.cuda()
                grapheme_logits, vowel_logits, consonant_logits = model(input_data)
                eval_result = evaluator(grapheme_logits, vowel_logits, consonant_logits, labels)
                eval_result = {k: eval_result[k].item() for k in eval_result}
                total_err += eval_result['loss']
                total_acc += eval_result['acc']
                # print(total_err / (1 + idx), total_acc / (1 + idx))

        val_total_err = total_err / (1 + idx)
        val_total_acc = total_acc / (1 + idx)
        print("Epoch {0} Eval, Loss {1}, Acc {2}".format(epoch, val_total_err, val_total_acc))

        print("Saving the model (epoch %d)" % epoch)
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, state_fpath)

        print("Making a backup (step %d)" % epoch)
        backup_fpath = os.path.join(backup_dir, "model_bak_%06d.pt" % (epoch,))
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, backup_fpath)

        perf_trace.append(
            {
                'epoch': epoch,
                'train_err': train_total_err,
                'train_acc': train_total_acc,
                'val_err': val_total_err,
                'val_acc': val_total_acc
            }
        )
        json.dump(perf_trace, open(perf_path, 'w'))
