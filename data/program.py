import pandas as pd
import numpy as np
import os 

def generate_dataset(n_samples=1000):
    """Generate synthetic social media account data"""
    np.random.seed(42)
    
    data = []
    
    for i in range(n_samples):
        # Randomly decide if account is fake (1) or real (0)
        is_fake = np.random.choice([0, 1], p=[0.6, 0.4])
        
        if is_fake:
            # Fake account characteristics
            profile_pic = np.random.choice([0, 1], p=[0.7, 0.3])  # Less likely to have profile pic
            num_followers = np.random.randint(0, 100)
            num_following = np.random.randint(500, 5000)
            num_posts = np.random.randint(0, 10)
            bio_length = np.random.randint(0, 20)
            account_age_days = np.random.randint(1, 60)
            has_url = np.random.choice([0, 1], p=[0.6, 0.4])
            avg_likes = np.random.randint(0, 5)
            avg_comments = np.random.randint(0, 2)
            username_length = np.random.randint(15, 30)
            has_numbers_in_username = np.random.choice([0, 1], p=[0.2, 0.8])
            follower_following_ratio = num_followers / (num_following + 1)
        else:
            # Real account characteristics
            profile_pic = np.random.choice([0, 1], p=[0.1, 0.9])  # More likely to have profile pic
            num_followers = np.random.randint(50, 10000)
            num_following = np.random.randint(50, 2000)
            num_posts = np.random.randint(10, 500)
            bio_length = np.random.randint(20, 150)
            account_age_days = np.random.randint(60, 2000)
            has_url = np.random.choice([0, 1], p=[0.4, 0.6])
            avg_likes = np.random.randint(10, 500)
            avg_comments = np.random.randint(2, 50)
            username_length = np.random.randint(5, 15)
            has_numbers_in_username = np.random.choice([0, 1], p=[0.7, 0.3])
            follower_following_ratio = num_followers / (num_following + 1)
        
        data.append({
            'profile_pic': profile_pic,
            'num_followers': num_followers,
            'num_following': num_following,
            'num_posts': num_posts,
            'bio_length': bio_length,
            'account_age_days': account_age_days,
            'has_url': has_url,
            'avg_likes': avg_likes,
            'avg_comments': avg_comments,
            'username_length': username_length,
            'has_numbers_in_username': has_numbers_in_username,
            'follower_following_ratio': round(follower_following_ratio, 4),
            'is_fake': is_fake
        })
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    # Generate dataset
    df = generate_dataset(1000)
    
    # Save to CSV
    output_path = os.path.join(os.path.dirname(__file__), 'social_media_accounts.csv')
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(f"Total samples: {len(df)}")
    print(f"Fake accounts: {df['is_fake'].sum()}")
    print(f"Real accounts: {len(df) - df['is_fake'].sum()}")