# YML=docker-compose.$(USER).yml
all:
	echo "Usage:"
	echo "  make image              # build image"
	echo "  make up                 # start container"
	echo "  make exec               # go into bash of container"
	echo "  make down               # stop all running services from docker-compose file"
	echo "  make mount FOLDER=dir   # mount folder 'dir' on /data/${USER}/diskstation/dir"
	echo "  make unmount FOLDER=dir # unmount folder 'dir'"

image:
	PORT=10710 \
	docker-compose -f docker-compose.yml -f docker-compose.${USER}.yml build

# Spin up the container
up:
	PORT=10710 \
	docker-compose -f docker-compose.yml -f docker-compose.${USER}.yml up -d

jupyter:
	docker exec -it riskawarerl jupyter lab --ip=0.0.0.0 --port=10710 --allow-root

# Start terminal in the container
exec:
	PORT=10710 \
	docker exec -it -e DISPLAY=${DISPLAY} riskawarerl bash
# Stop and remove containerm
down:
	PORT=10710 \
	docker-compose -f docker-compose.yml -f docker-compose.${USER}.yml down

# Restart container
restart:
	PORT=10710 \
	docker-compose -f docker-compose.yml -f docker-compose.${USER}.yml restart

# # Mount the share
# mount:
# 	mkdir /data/${USER}/diskstation/${FOLDER} || true
# 	smbmount diskstationii1 $(FOLDER) /data/${USER}/diskstation/${FOLDER}
# unmount:
# 	smbumount /data/${USER}/diskstation/${FOLDER}