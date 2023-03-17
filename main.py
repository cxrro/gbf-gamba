import numpy as np
# import pytesseract
import pyautogui
from collections import Counter
import itertools
import cv2
import random
import os
import glob

# Configure the path to the Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = '/System/Volumes/Data/opt/homebrew/Cellar/tesseract/5.3.0_1/bin/tesseract'


# Capture the screen
def capture_screen(region=None):
    screenshot = pyautogui.screenshot()
    if region:
        screenshot = screenshot.crop(region)
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)


# Load card value templates
def load_templates():
    templates = {}
    for file in glob.glob("templates/*.png"):
        name = os.path.splitext(os.path.basename(file))[0]
        templates[name] = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    return templates


# Match card value using template matching
def match_card_value(card_image, templates, threshold=0.9):
    card_gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
    for value, template in templates.items():
        res = cv2.matchTemplate(card_gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > threshold:
            return value
    return None

# Extract card values using template matching


def extract_card_values(card_images, templates):
    card_values = []
    for img in card_images:
        matched_value = match_card_value(img, templates)
        if matched_value is not None:
            card_values.append(matched_value)
        else:
            print("No match found for card value")
    return card_values


# Preprocess the image for contour detection
def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(
        gray_image, 128, 255, cv2.THRESH_BINARY)
    return thresholded_image


# Find card contours
def find_card_contours(image):
    preprocessed_image = preprocess_image(image)
    contours, _ = cv2.findContours(
        preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# Filter card contours based on size and aspect ratio
def filter_card_contours(contours, min_area=35000, max_area=45000, min_aspect_ratio=0.5, max_aspect_ratio=1):
    card_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / float(h)

        if min_area <= area <= max_area and min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
            card_contours.append((x, y, w, h))

    return card_contours

# Detect the positions of the playing cards


def detect_card_positions(screen):
    contours = find_card_contours(screen)
    card_positions = filter_card_contours(contours)
    return card_positions


""" def extract_card_values(card_images):
    card_values = []
    valid_values = ['2', '3', '4', '5', '6',
                    '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

    for img in card_images:
        # Apply OCR to the card image
        ocr_result = pytesseract.image_to_string(
            img, config='--psm 6 -c tessedit_char_whitelist=23456789TJQKA')

        # Check if OCR result is a valid card value
        if ocr_result.strip() in valid_values:
            card_values.append(ocr_result.strip())
        else:
            print(f"Invalid OCR result: {ocr_result.strip()}")

    return card_values """


def evaluate_poker_hand(hand):
    values = [v for v in hand if v]  # Ignore empty strings

    value_counts = Counter(values)

    value_ranks = {str(i): i for i in range(2, 11)}
    value_ranks.update({"10": 10, "J": 11, "Q": 12, "K": 13, "A": 14})

    sorted_values = sorted(values, key=lambda x: value_ranks[x])
    straight = all([value_ranks[sorted_values[i]] ==
                   value_ranks[sorted_values[i-1]] + 1 for i in range(1, len(sorted_values))])

    if straight:
        return 4
    elif any(count == 4 for count in value_counts.values()):
        return 20
    elif any(count == 3 for count in value_counts.values()):
        if any(count == 2 for count in value_counts.values()):  # Full house
            return 10
        else:  # 3 of a kind
            return 1
    elif sum(1 for count in value_counts.values() if count == 2) == 2:  # 2 pairs
        return 1
    return 0


def calculate_expected_value(kept_cards, discarded_cards):
    base_deck = ['2', '3', '4', '5', '6', '7',
                 '8', '9', '10', 'J', 'Q', 'K', 'A'] * 4

    # Make a copy of the base_deck before removing cards
    deck = base_deck.copy()
    for card in kept_cards:
        card_str = card
        if card_str not in deck:
            print(f"Card not found in deck: {card_str}")
            print(f"Kept cards: {kept_cards}")
            continue
        deck.remove(card_str)

    for card in kept_cards:
        card_str = card
        if card_str not in deck:
            print(f"Card not found in deck: {card_str}")
            print(f"Kept cards: {kept_cards}")
            continue
        deck.remove(card_str)

    num_simulations = 1000
    total_score = 0

    for _ in range(num_simulations):
        # Draw new cards for the discarded cards
        drawn_cards = random.sample(deck, len(discarded_cards))

        # Create the new hand
        new_hand = kept_cards + drawn_cards

        # Evaluate the new hand
        score = evaluate_poker_hand(new_hand)
        # print(f'hand: {new_hand} score: {score}')

        total_score += score

    # Calculate the average score (expected value) of the hand after mulligan
    expected_value = total_score / (num_simulations/100)
    return expected_value


def decide_mulligan(hand):
    best_score = -1
    best_combination = []

    # Iterate through all possible combinations of cards to keep (0 to 5 cards)
    for num_cards_to_keep in range(6):
        for combination in itertools.combinations(hand, num_cards_to_keep):
            kept_cards = list(combination)
            discarded_cards = [card for card in hand if card not in kept_cards]

            # Calculate the expected value of the current combination
            expected_value = calculate_expected_value(
                kept_cards, discarded_cards)

            # Print the current combination and its expected value
            print(
                f"Kept cards: {kept_cards}, Discarded cards: {discarded_cards}, Expected value: {expected_value}")

            # Update the best combination if the current expected value is higher
            if expected_value > best_score:
                best_score = expected_value
                best_combination = kept_cards

    # Get the indices of the cards to mulligan
    mulligan_indices = [i for i, card in enumerate(
        hand) if card not in best_combination]

    return mulligan_indices


# Main function
def main():

    # Capture the screen
    screen = capture_screen()
    cv2.imwrite('captured_screen.png', screen)  # Save the captured screen

    # Detect card positions
    card_positions = detect_card_positions(screen)
    print(f'Detected card positions: {card_positions}')

    # Sort card positions from left to right
    card_positions.sort(key=lambda pos: pos[0])

    # Extract card images and suits
    card_images = [screen[y:y+h, x:x+w] for x, y, w, h in card_positions]

    # Load the templates and extract card values
    templates = load_templates()
    card_values = extract_card_values(card_images, templates)
    print(f'Card values: {card_values}')

    # Draw bounding boxes around the detected cards
    for i, ((x, y, w, h), value) in enumerate(zip(card_positions, card_values)):
        cv2.rectangle(screen, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(screen, f"{value}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Save individual card images
        cv2.imwrite(f'card_{i}.png', card_images[i])

    # Save the image with bounding boxes
    cv2.imwrite('detected_cards.png', screen)

    if len(card_values) == 5:
        # Decide which cards to mulligan
        mulligan_indices = decide_mulligan(card_values)
        print(f'Mulligan indices: {mulligan_indices}')

        # Focus the window by clicking the top-left corner of the first card
        first_card_x, first_card_y, _, _ = card_positions[0]
        pyautogui.click(first_card_x/2, first_card_y/2)

        # Click on the cards to keep
        for index, (x, y, w, h) in enumerate(card_positions):
            if index not in mulligan_indices:
                # Calculate the click coordinates
                click_x = (x + w // 2)/2
                click_y = (y + h // 2)/2

                print(f"Click attempt at coordinates: ({click_x}, {click_y})")
                pyautogui.click(click_x, click_y)
    else:
        print("Not enough cards detected. Only found " + str(len(card_values)))


if __name__ == '__main__':
    main()
