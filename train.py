import schumt.trainer

if __name__ == "__main__":
    trainer = schumt.trainer.Trainer()
    for epoch in range(100):
        trainer.train_epoch()
