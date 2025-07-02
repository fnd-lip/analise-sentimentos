import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pygame
import cv2
import mediapipe as mp
import numpy as np
base_dir = os.path.dirname(__file__)
# Inicialização
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Jogo Rompe Blocos")
clock = pygame.time.Clock()
running = True
paused = True  

# Raquete
paddle_image = pygame.image.load(os.path.join(base_dir, 'assets', 'paddle.png'))
paddle_image = pygame.transform.scale(paddle_image, (150, 40))
paddle_x, paddle_y = WIDTH // 2, HEIGHT - 100
paddle_width, paddle_height = 150, 40

# Bola
ball_image = pygame.image.load(os.path.join(base_dir, 'assets', 'esfera.png'))
ball_image = pygame.transform.scale(ball_image, (40, 40))
ball_x, ball_y = WIDTH // 2, HEIGHT // 2
ball_speed_x, ball_speed_y = 8, 8
ball_radius = 20

# Blocos
block_rows, block_columns = 4, 8
block_width, block_height = WIDTH // block_columns, 50
block_img = pygame.image.load(os.path.join(base_dir, 'assets', 'tijolo.png'))
block_img = pygame.transform.scale(block_img, (block_width, block_height))

def create_blocks():
    return [pygame.Rect(col * block_width,
                        row * block_height,
                        block_width,
                        block_height)
            for row in range(block_rows)
            for col in range(block_columns)]

blocks = create_blocks()

# Função de reset
def reset_game():
    global ball_x, ball_y, ball_speed_x, ball_speed_y, blocks, paused
    ball_x, ball_y = WIDTH // 2, HEIGHT // 2
    ball_speed_x, ball_speed_y = 8, 8
    blocks[:] = create_blocks()
    paused = True  # Após reset, volta pausado
    print("Jogo resetado! Pressione P para começar.")

# Câmera
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    max_num_hands=1) as hands:
    while running:
        # Eventos de teclado
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    paused = not paused
                    if paused:
                        print("Jogo pausado.")
                    else:
                        print("Jogo iniciado/despausado.")
                elif event.key == pygame.K_r:
                    reset_game()

        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar vídeo da câmera.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # ✅ Controle da raquete SEMPRE ativo
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_finger = hand_landmarks.landmark[8]
                paddle_x = WIDTH - int(index_finger.x * WIDTH) - paddle_width // 2
                mp_draw.draw_landmarks(rgb_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Limitar raquete dentro da tela
        if paddle_x < 0:
            paddle_x = 0
        if paddle_x + paddle_width > WIDTH:
            paddle_x = WIDTH - paddle_width

        # ✅ Movimento da bola e colisões SÓ quando despausado
        if not paused:
            # Movimento da bola
            ball_x += ball_speed_x
            ball_y += ball_speed_y

            if ball_x - ball_radius < 0 or ball_x + ball_radius > WIDTH:
                ball_speed_x *= -1

            if ball_y - ball_radius < 0:
                ball_speed_y *= -1

            if ball_y + ball_radius > HEIGHT:
                print("Você perdeu! Pressione R para reiniciar.")
                paused = True  # Pausa se perder

            # Colisão com raquete
            if paddle_x < ball_x < paddle_x + paddle_width and ball_y + ball_radius >= paddle_y:
                ball_speed_y *= -1

            # Colisão com blocos
            for block in blocks:
                if block.collidepoint(ball_x, ball_y):
                    blocks.remove(block)
                    ball_speed_y *= -1
                    break

            # Vitória
            if not blocks:
                print("Você venceu! Pressione R para jogar novamente.")
                paused = True

        # Mostrar vídeo como fundo
        video_surface = np.rot90(cv2.resize(rgb_frame, (WIDTH, HEIGHT)))
        video_surface = pygame.surfarray.make_surface(video_surface)
        screen.blit(video_surface, (0, 0))

        # Desenhar elementos
        screen.blit(paddle_image, (paddle_x, paddle_y))
        screen.blit(ball_image, (ball_x - ball_radius, ball_y - ball_radius))
        for block in blocks:
            screen.blit(block_img, (block.x, block.y))

        # ✅ Tela de pausa com fundo verde
        if paused:
            overlay = pygame.Surface((WIDTH, HEIGHT))
            overlay.set_alpha(180)        # Transparência
            overlay.fill((0, 128, 0))     # Verde escuro
            screen.blit(overlay, (0, 0))

            font = pygame.font.SysFont(None, 80)
            text = font.render("PAUSADO", True, (255, 255, 255))
            rect = text.get_rect(center=(WIDTH//2, HEIGHT//2 - 50))
            screen.blit(text, rect)

            font2 = pygame.font.SysFont(None, 40)
            hint = font2.render("Pressione P para continuar | R para resetar", True, (255, 255, 255))
            rect2 = hint.get_rect(center=(WIDTH//2, HEIGHT//2 + 20))
            screen.blit(hint, rect2)

        pygame.display.flip()
        clock.tick(30)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
