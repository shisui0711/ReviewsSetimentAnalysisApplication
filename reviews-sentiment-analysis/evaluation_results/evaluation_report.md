
# Multilingual Sentiment Analysis Model Evaluation Report

## Overall Performance
- **Accuracy**: 0.7082
- **Macro F1 Score**: 0.6752
- **Weighted F1 Score**: 0.7190

## Per-Class Performance

### Negative Class
- Precision: 0.7966
- Recall: 0.7090
- F1 Score: 0.7503
- Support: 2000

### Neutral Class
- Precision: 0.3988
- Recall: 0.5340
- F1 Score: 0.4566
- Support: 1000

### Positive Class
- Precision: 0.8448
- Recall: 0.7945
- F1 Score: 0.8189
- Support: 2000

## Confidence Statistics
- Mean Confidence: 0.7384
- Confidence Std: 0.1334
- Min Confidence: 0.3589
- Max Confidence: 0.9232

## Sample Predictions

✓ **Text**: Don’t waste your time! These are AWFUL. They are see through, the fabric feels like tablecloth, and ...
   **True**: negative | **Predicted**: negative 
   **Confidence**: 0.744

✓ **Text**: One Star I bought 4 and NONE of them worked. Yes I used new batteries!
   **True**: negative | **Predicted**: negative 
   **Confidence**: 0.771

✓ **Text**: Totally useless On first use it didn't heat up and now it doesn't work at all
   **True**: negative | **Predicted**: negative 
   **Confidence**: 0.776

✓ **Text**: Gold filled earrings You want an HONEST answer? I just returned from UPS where I returned the FARCE ...
   **True**: negative | **Predicted**: negative 
   **Confidence**: 0.531

✗ **Text**: Poor container The glue works fine but the container is impossible to work with. The cap doesn't com...
   **True**: negative | **Predicted**: neutral 
   **Confidence**: 0.716

✓ **Text**: Won’t buy again I was over the moon when I got this. I love sunflowers and it looked exactly like th...
   **True**: negative | **Predicted**: negative 
   **Confidence**: 0.769

✗ **Text**: Very brittle case Gotta say, not impressed with the quality. I mean, I can't say I expected it to be...
   **True**: negative | **Predicted**: neutral 
   **Confidence**: 0.708

✓ **Text**: My 9 year old daughter was so disappointed. It didn't even have a different piece for ... I would gi...
   **True**: negative | **Predicted**: negative 
   **Confidence**: 0.726

✓ **Text**: Some of the glasses were broken when they arrived. Two of the glasses were broken when I opened the ...
   **True**: negative | **Predicted**: negative 
   **Confidence**: 0.616

✓ **Text**: Horrible Doesn’t even work . Did nothing for me :(
   **True**: negative | **Predicted**: negative 
   **Confidence**: 0.761
