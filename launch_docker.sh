#!/bin/bash

# Carlos Gómez-Huélamo - September 2020
# Modified Javier Araluce February 2021

# File to launch a docker image. Arguments:

# 1. Image name
# 2. Container name
# 3. User's name
# 4. Number of tabs
# 5. Force to kill the container associated to that image (if exists) and create a new one (true/false)

# Example: ./launch_docker_image.sh my_docker_image:last my_container_name my_user 4 false

# N.B. If this user was not previously included in the docker image, first enter as root and include this user:

	# ./launch_docker_image.sh my_docker_image:last my_container_name root

	# (Docker image) sudo useradd -m my_user (-m option creates the home directory for that user)
        # (Docker image) sudo passwd my_user -> (Enter your password for this user)
        
        # Open new host tab -> docker commit my_container my_image

        # Now you can use your new user (with its corresponding home directory)

# Function to create a new container and run multiple tabs

create_new_container()
{
	# Kill previous container

	docker stop $2
	docker rm -fv $2

	# Create and run multiple tabs

	if [[ $4 -gt 1 ]]; # Multiple tabs
	then
		for (( i=1; i<=$4; i++ ))
		do 
			command=""
			if [[ $i -eq 1 ]]; 
			then 
				command="docker run -it \
				--net host \
				"${FLAGS[@]}" \
				--name=$2 \
				-u $3 \
				-v /dev:/dev \
				--device /dev/snd \
				-e PULSE_SERVER=unix:${XDG_RUNTIME_DIR}/pulse/native \
				-v ${XDG_RUNTIME_DIR}/pulse/native:${XDG_RUNTIME_DIR}/pulse/native \
				--group-add $(getent group audio | cut -d: -f3) \
				-v /tmp/.X11-unix:/tmp/.X11-unix \
				-v $shared \
				-e DISPLAY=unix$DISPLAY $1 /bin/bash"
			else
				command="bash -c 'docker exec -it $2 /bin/bash'"
			fi

			gnome-terminal --tab "$i" -e "$command"
		done
	else 	
	# Single tab       
		docker run -it \
			--net host \
			"${FLAGS[@]}" \
			--name=$2 \
			-u $3 \
			-v /dev:/dev \
			--device /dev/snd \
			-e PULSE_SERVER=unix:${XDG_RUNTIME_DIR}/pulse/native \
			-v ${XDG_RUNTIME_DIR}/pulse/native:${XDG_RUNTIME_DIR}/pulse/native \
			--group-add $(getent group audio | cut -d: -f3) \
			-v /tmp/.X11-unix:/tmp/.X11-unix \
			-v $shared \
			-e DISPLAY=unix$DISPLAY $1 /bin/bash
	fi
}

# --runtime=nvidia

# Function to restart a stopped container and run multiple tabs
#GPU 0
#GPU-55a5e8e6-0b0a-a070-66a7-943b8056c017

#GPU 1
#GGPU-959effdb-d742-1ca6-e3d6-c48c6cdc023a

restart_container()
{
	if [[ $4 -gt 0 ]]; # Multiple tabs
	then
		docker start $2

		for (( i=1; i<=$4; i++ ))
		do 
			command="bash -c 'docker exec -it $2 bash'"

			gnome-terminal --tab "$i" -e "$command"
		done
	else 	           # Single tab
		docker start $2 && docker exec -it $2 bash
	fi
}

# 1. Set named volumes

shared=""
if [[ $3 != "root" ]]; # Non-root user
then
    shared=$HOME/ARAGAN:/home/$3/ARAGAN:Z
else                   # Root user
    shared=$HOME/ARAGAN:/$3/ARAGAN
fi

# 2. Check status of the container

check_running=$(docker ps --filter "name=$2" -q)
check_all=$(docker ps -a --filter "name=$2" -q)

echo "Image name: " $1
echo "Container name: " $2
echo "User: " $3
echo "Number of tabs: " $4

if [[ $6 -eq 0 ]]; 
then
	FLAGS=(--gpus '"device=0"')
elif [[ $6 -eq 1 ]]; 
then
	FLAGS=(--gpus '"device=1"')
elif [[ $6 -eq 2 ]]; 
then
	FLAGS=(--gpus '"device=2"')
elif [[ $6 -eq 3 ]]; 
then
	FLAGS=(--gpus '"device=3"')
elif [[ $6 -eq 4 ]]; 
then
	FLAGS=(--gpus '"device=4"')
elif [[ $6 -eq 5 ]]; 
then
	FLAGS=(--gpus '"device=5"')
elif [[ $6 -eq 6 ]]; 
then
	FLAGS=(--gpus '"device=6"')
elif [[ $6 -eq 7 ]]; 
then
	FLAGS=(--gpus '"device=7"')
else
	FLAGS=(--gpus all --privileged)
fi

echo "${FLAGS[@]}"

# 3. Run the image

if [[ -z $check_all ]]; 
then
	echo $'\nThe container does not exist'

	# 3.1. Create new container 

	create_new_container $1 $2 $3 $4
else
	# 3.2. Check if your container is stopped or it is currently running

	if [[ -z "$check_running" ]];
	then
		echo $'\nThe container is stopped. Restart and run multiple tabs or create a new one'

		if [[ $5 == "true" ]];
		then
			echo "Create a new container"
			create_new_container $1 $2 $3 $4
		else
			echo "Restart the container"
			restart_container $1 $2 $3 $4
		fi
	else	
		echo $'\nThe container is currently running'
	
		if [[ $5 == "true" ]];
		then
			echo "Create a new container"
			create_new_container $1 $2 $3 $4
		else
			echo "Restart the container"
			restart_container $1 $2 $3 $4
		fi
		
		# TODO #

		# Check the number of tabs of the container. If it is running, but the number of tabs is lower than the specified
		# number in the CLI (Command Line Interface), create the remaining number of tabs

		# Function to create a new container and run multiple tabs
	fi
fi

# --runtime=nvidia
