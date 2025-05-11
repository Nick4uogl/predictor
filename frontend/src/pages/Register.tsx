import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Box, Container, TextField, Button, Typography, Paper } from '@mui/material';
import { useAuth } from '../context/AuthContext';

const Register: React.FC = () => {
  const navigate = useNavigate();
  const { register } = useAuth();
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [errors, setErrors] = useState<{
    name?: string;
    email?: string;
    password?: string;
    general?: string;
  }>({});

  const validateForm = () => {
    const newErrors: typeof errors = {};

    if (!name.trim()) {
      newErrors.name = "Ім'я обов'язкове";
    }

    if (!email.trim()) {
      newErrors.email = "Email обов'язковий";
    } else if (!/\S+@\S+\.\S+/.test(email)) {
      newErrors.email = "Введіть коректну email адресу";
    }

    if (!password) {
      newErrors.password = "Пароль обов'язковий";
    } else if (password.length < 8) {
      newErrors.password = "Пароль повинен містити не менше 8 символів";
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrors({});

    if (!validateForm()) {
      return;
    }

    try {
      await register(email, password, name);
      navigate('/');
    } catch (error) {
      if (error instanceof Error) {
        const errorMessage = error.message;
        
        if (errorMessage.includes('400')) {
          // Check for specific email validation error
          if (errorMessage.includes('Некоректна електронна адреса')) {
            setErrors({ email: "Введіть коректну email адресу" });
          } else {
            setErrors({ general: "Перевірте правильність введених даних" });
          }
        } else if (errorMessage.includes('409')) {
          setErrors({ email: "Користувач з такою email адресою вже існує" });
        } else {
          setErrors({ general: "Помилка під час реєстрації. Спробуйте пізніше." });
        }
      }
    }
  };

  return (
    <Container maxWidth="sm">
      <Box sx={{ mt: 8, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <Paper elevation={3} sx={{ p: 4, width: '100%' }}>
          <Typography component="h1" variant="h5" align="center" gutterBottom>
            Реєстрація
          </Typography>
          {errors.general && (
            <Typography color="error" align="center" sx={{ mb: 2 }}>
              {errors.general}
            </Typography>
          )}
          <Box component="form" onSubmit={handleSubmit} sx={{ mt: 1 }}>
            <TextField
              margin="normal"
              required
              fullWidth
              id="name"
              label="Ім'я"
              name="name"
              autoComplete="name"
              autoFocus
              value={name}
              onChange={(e) => setName(e.target.value)}
              error={!!errors.name}
              helperText={errors.name}
            />
            <TextField
              margin="normal"
              required
              fullWidth
              id="email"
              label="Email"
              name="email"
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              error={!!errors.email}
              helperText={errors.email}
            />
            <TextField
              margin="normal"
              required
              fullWidth
              name="password"
              label="Пароль"
              type="password"
              id="password"
              autoComplete="new-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              error={!!errors.password}
              helperText={errors.password}
            />
            <Button
              type="submit"
              fullWidth
              variant="contained"
              sx={{ mt: 3, mb: 2 }}
            >
              Зареєструватися
            </Button>
          </Box>
        </Paper>
      </Box>
    </Container>
  );
};

export default Register;
