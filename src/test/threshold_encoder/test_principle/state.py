def state_to_value(c1, c2, c3):
    """
    3つのチャンネルを線形和で連続値として表現し、-1～1の範囲に収める関数
    c1, c2, c3: それぞれ0または1の値
    """
    # 線形和で0～7の範囲に変換
    linear_value = c1 * 4 + c2 * 2 + c3
    # 0～7を-1～1にスケーリング
    scaled_value = (linear_value / 7.0) * 2 - 1
    return scaled_value

# テスト
states = [
    (0, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 1),
    (1, 0, 1),
    (1, 1, 0)
]

for state in states:
    print(f"State {state} -> Value {state_to_value(*state)}")