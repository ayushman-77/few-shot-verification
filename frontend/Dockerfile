FROM node:18-alpine

WORKDIR /app

# Install dependencies first for Docker caching
COPY package*.json ./
RUN npm install

# Copy all project files
COPY . .

# Expose the Vite port
EXPOSE 5173

# Command to run the development server and expose it
CMD ["npm", "run", "dev", "--", "--host"]
