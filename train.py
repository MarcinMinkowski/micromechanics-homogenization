from trainer import Trainer

if __name__ == "__main__":
    
    trainer = Trainer()
    trainer.train_loop(5)
    print(f"Test loss: {trainer.test_loop()}")
    trainer.save_weights()