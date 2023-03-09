import time
from utils_i2u import *       #changed
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu

def train(train_loader, model, criterion, optimizer, epoch, writer, grad_clip, device,
          print_freq = 100, kl_weight = 0):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param model: the model to be trained
    :param criterion: loss layer
    :param optimizer: 
    :param epoch: epoch number
    :param writer: TF's logger. Writes values to tensorboard.
    :param grad_clip: clip gradients larger than give value.
    :param device: the device where the model is. ("cpu" or "cuda")
    """
    model.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens, padding_mask) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
        caplens = caplens.squeeze()
        padding_mask = padding_mask.to(device)

        # Forward prop.
        if kl_weight is None:
            logits, encoded_seq, decode_lengths, sort_ind = model(
                imgs,
                caps,
                padding_mask,
                caplens
            )
        else:
            logits, encoded_seq, decode_lengths, sort_ind, kl_loss = model(
                imgs,
                caps,
                padding_mask,
                caplens
            )
        
        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = encoded_seq[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores, _, _, _ = pack_padded_sequence(logits, decode_lengths, batch_first=True, enforce_sorted=False)
        targets, _, _, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True, enforce_sorted=False)

        # Calculate loss
        # if kl_weight is None:
        if kl_weight is None:
            loss = criterion(scores, targets)
        else:
            loss = criterion(scores, targets) + kl_weight*kl_loss

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Grad Clip preveting grad explosion
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        optimizer.step()
        # this_accuracy = torch.sum(torch.argmax(scores, dim=1) == targets) / logits.size(dim=0)
        
        # Keep tracks of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        if i % 10 == 0:
            niter = epoch * len(train_loader) + i
            writer.add_scalar('Train/Loss', loss.data, niter)
            writer.add_scalar('Train/Top-5-Accuracy', top5, niter)

        start = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))

def validate_beam(val_loader, model, start_unit, end_unit, epoch, writer, decode_num, device):
    """
        Bleu-4 Score should not be computed by decoded seqs from GT ans.
        It should compute between generated sequence (by beam-search or other generative decoding)
        and the GT seqs.
    """
    model.eval()
    # batch_time = AverageMeter()
    # We don't have loss or accuracy if we use beam search to decode.
    # losses = AverageMeter()
    # top5accs = AverageMeter()
    # bleu_4 = AverageMeter()
    start = time.time()

    # start_unit = word_map["<start>"]
    # end_unit = word_map["<end>"]

    refs = list() # GT captions
    hypos = list() # pred captions
    zero_count = 0
    for i, (imgs, caps, caplens, padding_mask, all_caps, all_padding_mask) in enumerate(iter(val_loader)):
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
        caplens = caplens.squeeze()
        padding_mask = padding_mask.to(device)
        all_caps = all_caps.to(device)
        all_padding_mask = all_padding_mask.to(device)

        pred_seq = model.decode(
            start_unit = start_unit,
            end_unit = end_unit,
            image = imgs,
            max_len = 150,
            beam_size = 5,
        )

        pred_seq = [w for w in pred_seq if w not in {start_unit, end_unit}]
        if len(pred_seq) == 0:
            zero_count += 1
        if i == 5 and zero_count >= 3:
            return 0

        hypos.append(pred_seq)
        # hypos

        for j in range(all_caps.shape[0]):
            img_caps = all_caps[j].tolist()
            img_captions = list(
                map(lambda c: [w for w in c if w not in {start_unit, end_unit}],
                    img_caps))  # remove <start> and pads
            refs.append(img_captions)
        assert len(hypos) == len(refs)
        
        if i >= decode_num:
            break

    bleu_4 = corpus_bleu(refs, hypos)
    writer.add_scalar('Valid/Bleu4', bleu_4, epoch)

    print(f"Valid Time: {time.time() - start}")
    return bleu_4

def validate(val_loader, model, criterion, epoch, writer, device, start_unit, end_unit,
          print_freq = 100, kl_weight = 0): # -> bleu-4, accuracy
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()
    start = time.time()

    refs = list() # GT captions
    hypos = list() # pred captions

    with torch.no_grad():
        for i, (imgs, caps, caplens, padding_mask, all_caps, all_padding_mask) in enumerate(val_loader):
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            caplens = caplens.squeeze()
            padding_mask = padding_mask.to(device)
            all_caps = all_caps.to(device)
            all_padding_mask = all_padding_mask.to(device)

            if kl_weight is None:
                logits, encoded_seq, decode_lengths, sort_ind = model(
                    imgs,
                    caps,
                    padding_mask,
                    caplens
                )
            else:
                logits, encoded_seq, decode_lengths, sort_ind, kl_loss = model(
                    imgs,
                    caps,
                    padding_mask,
                    caplens
                )

            targets = encoded_seq[:, 1:]
            scores_copy = logits.clone()
            scores, _, _, _ = pack_padded_sequence(logits, decode_lengths, batch_first=True, enforce_sorted=False)
            targets, _, _, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True, enforce_sorted=False)
            
            if kl_weight is None:
                loss = criterion(scores, targets)
            else:
                loss = criterion(scores, targets) + kl_weight*kl_loss
            # Keep track of current validations
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)
            
            # niter = epoch * len(train_loader) + i
            # writer.add_scalar('Train/Loss', loss.data, niter)
            # writer.add_scalar('Train/Top-5-Accuracy', top5, niter)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            all_caps = all_caps[sort_ind]  # because images were sorted in the decoder
            for j in range(all_caps.shape[0]):
                img_caps = all_caps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {start_unit, end_unit}],
                        img_caps))  # remove <start> and pads
                refs.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypos.extend(preds)

            assert len(refs) == len(hypos)
        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(refs, hypos)

        # Write to tensorboard
        # niter = epoch
        # writer.add_scalar('Valid/Loss', losses.avg, niter)
        # writer.add_scalar('Valid/Top-5-Accuracy', top5accs.avg, niter)
        # writer.add_scalar('Valid/BLEU-4', bleu4, niter)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))
        
        writer.add_scalar("Valid/Bleu4", bleu4, epoch)
        writer.add_scalar("Valid/Top5Accuracy", top5accs.avg, epoch)
        writer.add_scalar("Valid/Losses", losses.avg, epoch)
    return bleu4, top5accs.avg, losses.avg


def train_LM(train_loader, model, criterion, optimizer, epoch, writer, grad_clip, device,
          print_freq = 100):
    
    model.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy
    perplexity = AverageMeter()

    start = time.time()
    
    # This codes are for debug: check the model params works
    # for tag, value in model.named_parameters():
    #     tag = tag.replace('.', '/')
    #     print(tag, value)

    # add perplextiy
    # pp = Perplexity()

    # Batches
    for i, (caps, caplens, padding_mask) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        caps = caps.to(device)
        caplens = caplens.to(device)
        caplens = caplens.squeeze()
        padding_mask = padding_mask.to(device)

        # Forward prop.
        logits, encoded_seq, decode_lengths, sort_ind = model(
            caps,
            padding_mask,
            caplens
        )

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = encoded_seq[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        # pp.update(logits, targets)
        scores, _, _, _ = pack_padded_sequence(logits, decode_lengths, batch_first=True, enforce_sorted=False)
        targets, _, _, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True, enforce_sorted=False)

        # Calculate loss
        loss = criterion(scores, targets)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Grad Clip preveting grad explosion
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        optimizer.step()
        # this_accuracy = torch.sum(torch.argmax(scores, dim=1) == targets) / logits.size(dim=0)
        
        # Keep tracks of metrics
        # add Perplexity
        pp = torch.exp(loss)
        perplexity.update(pp, sum(decode_lengths))

        # top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        # top5accs.update(top5, sum(decode_lengths))

        batch_time.update(time.time() - start)

        if i % 10 == 0:
            niter = epoch * len(train_loader) + i
            writer.add_scalar('Train/Loss', loss.data, niter)
            #writer.add_scalar('Train/Avg_Perplexity', top5, niter)
            writer.add_scalar('Train/Avg_Perplexity', pp, niter)

        start = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Perplexity {perplexity.val:.3f} ({perplexity.avg:.3f}) '.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          perplexity=perplexity))

def validate_LM(val_loader, model, criterion, epoch, writer, device, print_freq = 100): # -> bleu-4, accuracy
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    perplexity = AverageMeter()
    start = time.time()

    # add perplexity
    # pp = Perplexity()

    with torch.no_grad():
        for i, (caps, caplens, padding_mask) in enumerate(val_loader):

            caps = caps.to(device)
            caplens = caplens.to(device)
            caplens = caplens.squeeze()
            padding_mask = padding_mask.to(device)

            logits, encoded_seq, decode_lengths, sort_ind = model(
                caps,
                padding_mask,
                caplens
            )

            targets = encoded_seq[:, 1:]
            scores_copy = logits.clone()
            scores, _, _, _ = pack_padded_sequence(logits, decode_lengths, batch_first=True, enforce_sorted=False)
            targets, _, _, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True, enforce_sorted=False)
            
            loss = criterion(scores, targets)

            # Keep track of current validations
            losses.update(loss.item(), sum(decode_lengths))
            # top5 = accuracy(scores, targets, 5)
            # top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            # add perplexity
            # pp.update(scores, targets)
            # pp_value = pp.compute()
            pp = torch.exp(loss)
            perplexity.update(pp, sum(decode_lengths))
            
            # niter = epoch * len(train_loader) + i
            # writer.add_scalar('Train/Loss', loss.data, niter)
            # writer.add_scalar('Train/Top-5-Accuracy', top5, niter)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Perplexity {perplexity.val:.3f} ({perplexity.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, perplexity=perplexity))
        print(
            '\n * LOSS - {loss.avg:.3f}, PERPLEXITY - {perplexity.avg:.3f}\n'.format(
                loss=losses,
                perplexity=perplexity))
        
        writer.add_scalar("Valid/Perplexity", perplexity.avg, epoch)
        writer.add_scalar("Valid/Losses", losses.avg, epoch)
    return perplexity.avg, losses.avg